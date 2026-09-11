"""Automatic food/drink region proposals using YOLO26 nano segmentation."""
import hashlib
import os

import numpy as np
from PIL import ImageDraw

from image_search import ROOT

DETECTOR = 'yolo26n-seg.pt'
WEIGHTS = ROOT / '.cache/yolo' / DETECTOR
FOOD_CLASSES = {'bottle', 'wine glass', 'cup', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'}


def filter_boxes(boxes, scores, labels, size, threshold=.25, iou_threshold=.5, min_area=.01, polygons=None):
    """Clamp and suppress duplicate boxes without using any ground-truth labels."""
    w, h = size
    candidates = []
    for index, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        l, t, r, b = box
        box = [max(0, int(np.floor(l))), max(0, int(np.floor(t))),
               min(w, int(np.ceil(r))), min(h, int(np.ceil(b)))]
        l, t, r, b = box
        if score >= threshold and r > l and b > t and (r-l)*(b-t) >= w*h*min_area:
            candidate = {'box': box, 'score': float(score), 'label': str(label)}
            if polygons is not None:
                candidate['polygon'] = polygons[index]
            candidates.append(candidate)
    kept = []
    for c in sorted(candidates, key=lambda c: -c['score']):
        a = c['box']
        duplicate = False
        for other in kept:
            b = other['box']
            intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
            union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
            if intersection / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(c)
    return kept


class FoodDetector:
    def __init__(self):
        (ROOT / '.cache/ultralytics/Ultralytics').mkdir(parents=True, exist_ok=True)
        (ROOT / '.cache/matplotlib').mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.cache/ultralytics'))
        os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
        from ultralytics import YOLO
        import torch
        torch.set_num_threads(4)
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(WEIGHTS))
        self.revision = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
        self.classes = [i for i, name in self.model.names.items() if name in FOOD_CLASSES]
        if not self.classes:
            raise ValueError('Checkpoint does not expose supported food/container classes')

    def detect(self, image):
        # Integer class IDs filter fixed pretrained categories; there is no text encoder or prompt.
        result = self.model.predict(image, device='cpu', imgsz=640, conf=.25,
                                    classes=self.classes, verbose=False, retina_masks=True)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        if result.masks is None:
            raise ValueError('Expected an instance-segmentation checkpoint')
        return filter_boxes(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                            [result.names[int(i)] for i in result.boxes.cls.cpu().tolist()],
                            image.size, polygons=[p.tolist() for p in result.masks.xy])


def annotate(image, detections):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for i, item in enumerate(detections, 1):
        if len(item.get('polygon', [])) >= 3:
            draw.polygon([tuple(p) for p in item['polygon']], outline='#ffaa00', width=2)
        draw.rectangle(item['box'], outline='#00dd88', width=3)
        draw.text((item['box'][0]+3, item['box'][1]+3), f"{i}: {item['label']} ({item['score']:.2f})", fill='black', stroke_width=1, stroke_fill='white')
    return image
