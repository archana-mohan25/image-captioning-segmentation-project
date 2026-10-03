"""Small NLP helpers: keyword extraction and a text summary that merges caption + segments."""
import re

STOPWORDS = set("""a an the and or of in on at to with by for from is are was were be being been it its this that
these those there their his her he she they them as into onto over under near next up down out off very
two three one some many other another has have having while holding sitting standing looking""".split())


def keywords(caption):
    words = re.findall(r"[a-zA-Z]+", caption.lower())
    seen, result = set(), []
    for w in words:
        if w not in STOPWORDS and len(w) > 2 and w not in seen:
            seen.add(w)
            result.append(w)
    return result


def scene_summary(caption, objects):
    if not objects:
        return f"{caption} No known object classes were segmented."
    parts = ", ".join(f"{o['label']} ({o['percent']}%)" for o in objects[:5])
    return f"{caption} Segmentation found: {parts}."
