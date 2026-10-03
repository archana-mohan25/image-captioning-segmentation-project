"""Image captioning with BLIP (vision encoder + transformer text decoder)."""
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"


class Captioner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(self.device).eval()

    @torch.no_grad()
    def caption(self, image, num_beams=4):
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        ids = self.model.generate(**inputs, max_new_tokens=30, num_beams=num_beams)
        text = self.processor.decode(ids[0], skip_special_tokens=True).strip()
        return text[:1].upper() + text[1:] + "."
