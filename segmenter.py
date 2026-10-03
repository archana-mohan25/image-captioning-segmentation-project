"""Semantic segmentation with a pretrained DeepLabV3 (ResNet-50) CNN."""
import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


class Segmenter:
    def __init__(self):
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = deeplabv3_resnet50(weights=weights).to(self.device).eval()
        self.transform = weights.transforms()
        self.classes = weights.meta["categories"]  # 21 Pascal VOC classes
        rng = np.random.RandomState(7)
        self.palette = rng.randint(60, 255, (len(self.classes), 3)).astype(np.uint8)
        self.palette[0] = 0  # background

    @torch.no_grad()
    def segment(self, image):
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)["out"][0]
        small = logits.argmax(0).cpu().numpy().astype(np.uint8)
        mask = np.array(Image.fromarray(small).resize(image.size, Image.NEAREST))

        color = self.palette[mask]
        base = np.array(image).astype(np.float32)
        fg = (mask > 0)[..., None]
        overlay = np.where(fg, 0.5 * base + 0.5 * color, base).astype(np.uint8)

        total = mask.size
        objects = []
        for cid in np.unique(mask):
            if cid == 0:
                continue
            objects.append({
                "label": self.classes[cid],
                "percent": round(float((mask == cid).sum()) * 100 / total, 1),
                "color": "#%02x%02x%02x" % tuple(int(v) for v in self.palette[cid]),
            })
        objects.sort(key=lambda o: -o["percent"])
        return Image.fromarray(overlay), Image.fromarray(color), objects
