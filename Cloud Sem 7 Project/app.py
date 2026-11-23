import io
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = FastAPI(title="MSAG-ResNet Diagnostics")


# ==========================================
# 1. ARCHITECTURE BLUEPRINTS
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)
        )
        self.out_conv = nn.Conv2d(out_channels * 4, out_channels, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_concat = torch.cat([
            self.relu(self.conv1(x)),
            self.relu(self.conv2(x)),
            self.relu(self.conv3(x)),
            F.interpolate(self.relu(self.global_avg_pool(x)), size=x.size()[2:], mode='bilinear', align_corners=True)
        ], dim=1)
        return self.relu(self.out_conv(x_concat))


class MSAG_ResNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.aspp = ASPP(512, 256)
        self.cbam = CBAM(256)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.aspp(x)
        x = self.cbam(x)
        return self.classifier(x)


# ==========================================
# 2. INITIALIZATION & PREPROCESSING
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MSAG_ResNet(num_classes=4)
model.load_state_dict(torch.load('msag_resnet.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
CLASS_NAMES = ['COVID-19', 'Lung Opacity', 'Normal', 'Viral Pneumonia']


def apply_clahe(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img)
    img_rgb = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_rgb), img_rgb


# ==========================================
# 3. FASTAPI ENDPOINTS
# ==========================================
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    pil_img, rgb_img = apply_clahe(img_bytes)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # A. MC Dropout (Epistemic Uncertainty)
    model.eval()  # CRITICAL FIX: Keep BatchNorm locked to trained statistics

    # Helper function to ONLY enable dropout layers, avoiding the BatchNorm bug
    def enable_dropout(m):
        if type(m) == nn.Dropout:
            m.train()

    model.apply(enable_dropout)

    mc_preds = []
    with torch.no_grad():
        for _ in range(30):
            mc_preds.append(F.softmax(model(input_tensor), dim=1).cpu().numpy())

    mc_preds = np.array(mc_preds)
    mean_probs = mc_preds.mean(axis=0)[0]
    variance = mc_preds.var(axis=0)[0]
    pred_idx = np.argmax(mean_probs)
    max_variance = np.max(variance)
    ##uncertainty_flag = bool(max_variance > 0.05)
    uncertainty_flag = bool(max_variance > 0.01 or mean_probs[pred_idx] < 0.75)

    # B. Grad-CAM Explainability
    model.eval()  # Turn dropout back off for the clean Grad-CAM generation

    targets = [ClassifierOutputTarget(pred_idx)]
    cam = GradCAM(model=model, target_layers=[model.cbam])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    rgb_normalized = cv2.resize(rgb_img, (224, 224)) / 255.0
    cam_image = show_cam_on_image(np.float32(rgb_normalized), grayscale_cam, use_rgb=True)

    _, buffer = cv2.imencode('.png', cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    gradcam_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": float(mean_probs[pred_idx] * 100),
        "uncertainty_flag": uncertainty_flag,
        "variance": float(max_variance),
        "gradcam_base64": f"data:image/png;base64,{gradcam_b64}"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)