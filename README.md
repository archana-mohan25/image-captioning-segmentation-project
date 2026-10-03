# Image Captioning and Segmentation

A Flask web app that takes an image and returns:
1. **A caption** (BLIP: vision transformer encoder + text decoder, the NLP part)
2. **Segmentation** (DeepLabV3 ResNet-50 CNN, 21 Pascal VOC classes)
3. **NLP extras**: keywords from the caption and a combined scene summary

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

The first run downloads the pretrained models (about 1 GB) and needs internet. After that it works offline. A GPU is optional; CPU takes a few seconds per image.

## Files
- `app.py`: Flask server and `/analyze` endpoint
- `captioner.py`: BLIP caption model
- `segmenter.py`: DeepLabV3 segmentation, overlay and mask
- `nlp.py`: keyword extraction and scene summary
- `templates/index.html`: the web page

## Notes
- Segmentation recognizes 21 classes (person, dog, car, chair, bottle and so on). Other objects are shown as background.
- To use your own trained CNN+LSTM captioner or U-Net, replace the classes in `captioner.py` / `segmenter.py` but keep the same `caption()` / `segment()` methods.
