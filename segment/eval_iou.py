"""Pixel-level and instance-level IoU, precision and recall for YOLOv5 segmentation.

Usage, from the repo root:
    python segment/eval_iou.py --weights best.pt --data data.yaml --device 0
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.dataloaders import IMG_FORMATS, img2label_paths  # noqa: E402
from utils.general import check_dataset, check_img_size, non_max_suppression  # noqa: E402
from utils.segment.general import process_mask  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


def list_images(val):
    files = []
    for entry in val if isinstance(val, list) else [val]:
        p = Path(entry)
        if p.is_dir():
            files += [f for f in p.rglob('*') if f.suffix[1:].lower() in IMG_FORMATS]
        else:
            files += [p.parent / line.strip() for line in p.read_text().splitlines() if line.strip()]
    return sorted(files)


# Masks are stored as (box, crop): box = x0, y0, x1, y1 in original-image pixels (x1, y1 exclusive), crop = the
# boolean mask inside that box. All pixels outside the box are 0. This gives exactly the same numbers as full-image
# masks, but memory and time grow with object size instead of image size times object count.


def load_gt(label_path, h, w):
    """Return class ids (n,), boxes (n, 4) and cropped boolean masks from a YOLO polygon label file."""
    classes, boxes, crops = [], [], []
    path = Path(label_path)
    if path.exists():
        for line in path.read_text().splitlines():
            values = line.split()
            if len(values) < 7:  # class + at least 3 points
                continue
            pts = np.round(np.array(values[1:], dtype=np.float64).reshape(-1, 2) * (w, h)).astype(np.int32)
            x0, y0 = max(int(pts[:, 0].min()), 0), max(int(pts[:, 1].min()), 0)
            x1, y1 = min(int(pts[:, 0].max()) + 1, w), min(int(pts[:, 1].max()) + 1, h)
            if x1 <= x0 or y1 <= y0:  # polygon fully outside the image: empty mask, still a ground-truth object
                x0 = y0 = x1 = y1 = 0
                crop = np.zeros((0, 0), dtype=bool)
            else:
                crop = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                cv2.fillPoly(crop, [(pts - np.array([x0, y0], dtype=np.int32)).astype(np.int32)], 1)
                crop = crop.astype(bool)
            classes.append(int(float(values[0])))
            boxes.append((x0, y0, x1, y1))
            crops.append(crop)
    return np.array(classes, dtype=int), np.array(boxes, dtype=int).reshape(-1, 4), crops


def nearest_index(n_out, n_in):
    """Source index of each output pixel for torch 'nearest' resize, with the same float32 math as PyTorch."""
    if n_out == n_in:
        return np.arange(n_out)
    scale = np.float32(n_in) / np.float32(n_out)
    return np.minimum(np.floor(np.arange(n_out, dtype=np.float32) * scale).astype(np.int64), n_in - 1)


def to_original(masks, pad, h, w):
    """Remove letterbox padding and map masks (n, H, W) to the original image size.

    Same pixels as F.interpolate(..., (h, w), mode='nearest') on the full masks, but only the area around each
    mask is built. Returns boxes (n, 4) and cropped boolean masks.
    """
    if not len(masks):
        return np.zeros((0, 4), dtype=int), []
    H, W = masks.shape[1:]
    top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))
    bottom, right = H - int(round(pad[1] + 0.1)), W - int(round(pad[0] + 0.1))
    m = masks[:, top:bottom, left:right].bool().cpu().numpy()  # (n, h', w') at letterbox scale, small
    rows, cols = nearest_index(h, m.shape[1]), nearest_index(w, m.shape[2])  # non-decreasing
    boxes, crops = [], []
    for mi in m:
        ys, xs = np.flatnonzero(mi.any(1)), np.flatnonzero(mi.any(0))
        if not len(ys):
            boxes.append((0, 0, 0, 0))
            crops.append(np.zeros((0, 0), dtype=bool))
            continue
        # output pixels whose source row/column lies inside the mask's extent
        y0, y1 = int(np.searchsorted(rows, ys[0], 'left')), int(np.searchsorted(rows, ys[-1], 'right'))
        x0, x1 = int(np.searchsorted(cols, xs[0], 'left')), int(np.searchsorted(cols, xs[-1], 'right'))
        boxes.append((x0, y0, x1, y1))
        crops.append(mi[rows[y0:y1]][:, cols[x0:x1]])
    return np.array(boxes, dtype=int).reshape(-1, 4), crops


def intersection(box_a, crop_a, box_b, crop_b):
    """Number of pixels set in both cropped masks."""
    x0, y0 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x1, y1 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    a = crop_a[y0 - box_a[1]:y1 - box_a[1], x0 - box_a[0]:x1 - box_a[0]]
    b = crop_b[y0 - box_b[1]:y1 - box_b[1], x0 - box_b[0]:x1 - box_b[0]]
    return int((a & b).sum())


def instance_stats(pred_cls, pred, gt_cls, gt, iou_thres):
    """Greedy matching in descending confidence order. Predictions must already be sorted.

    pred and gt are (boxes, crops) tuples from to_original() and load_gt().
    """
    (pred_boxes, pred_crops), (gt_boxes, gt_crops) = pred, gt
    pred_area = [int(c.sum()) for c in pred_crops]
    gt_area = [int(c.sum()) for c in gt_crops]
    used = np.zeros(len(gt_cls), dtype=bool)
    matched_ious = []
    fp = 0
    for k, cls in enumerate(pred_cls):
        best_iou, best_j = 0.0, -1
        for j in np.flatnonzero((gt_cls == cls) & ~used):
            inter = intersection(pred_boxes[k], pred_crops[k], gt_boxes[j], gt_crops[j])
            union = pred_area[k] + gt_area[j] - inter
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thres:
            used[best_j] = True
            matched_ious.append(float(best_iou))
        else:
            fp += 1
    return matched_ious, fp, int((~used).sum())


def paint(cls_ids, boxes, crops, c, h, w):
    """Union of all class-c cropped masks as one full-image boolean mask."""
    canvas = np.zeros((h, w), dtype=bool)
    for k in np.flatnonzero(cls_ids == c):
        x0, y0, x1, y1 = boxes[k]
        canvas[y0:y1, x0:x1] |= crops[k]
    return canvas


def pixel_stats(pred_cls, pred, gt_cls, gt, classes, h, w):
    """Per-class IoU, precision and recall for one image, using the union of each class's masks."""
    ious, precisions, recalls = [], [], []
    for c in classes:
        if not (pred_cls == c).any() and not (gt_cls == c).any():
            continue  # class absent and correctly not predicted
        pred_c = paint(pred_cls, *pred, c, h, w)
        gt_c = paint(gt_cls, *gt, c, h, w)
        n_pred, n_gt = int(pred_c.sum()), int(gt_c.sum())
        if not n_pred and not n_gt:
            continue  # class absent and correctly not predicted
        tp = int((pred_c & gt_c).sum())
        ious.append(tp / (n_pred + n_gt - tp))
        if n_pred:
            precisions.append(tp / n_pred)
        if n_gt:
            recalls.append(tp / n_gt)
    return ious, precisions, recalls


def mean(values):
    return float(np.mean(values)) if len(values) else float('nan')


@torch.no_grad()
def main(opt):
    device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=device, data=opt.data)
    stride, names = model.stride, model.names
    imgsz = check_img_size(opt.imgsz, s=stride)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))

    data = check_dataset(opt.data)
    files = list_images(data[opt.split])
    labels = img2label_paths([str(f) for f in files])
    classes = list(range(len(names)))
    if not files:
        raise SystemExit(f'No images found for split {opt.split}')

    px_iou, px_precision, px_recall, matched = [], [], [], []
    tp = fp = fn = 0
    for i, (file, label) in enumerate(zip(files, labels), 1):
        im0 = cv2.imread(str(file))
        if im0 is None:
            print(f'WARNING: cannot read {file}')
            continue
        h, w = im0.shape[:2]
        im, _, pad = letterbox(im0, imgsz, stride=stride, auto=model.pt)
        im = torch.from_numpy(im.transpose(2, 0, 1)[::-1].copy()).to(device).float()[None] / 255
        out, proto = model(im)[:2]
        det = non_max_suppression(out, opt.conf_thres, opt.iou_nms, max_det=1000, nm=proto.shape[1])[0]
        masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True) if len(det) else det[:0, 0:0]
        pred = to_original(masks, pad, h, w)
        pred_cls = det[:, 5].int().cpu().numpy()
        gt_cls, *gt = load_gt(label, h, w)

        ious, fp_i, fn_i = instance_stats(pred_cls, pred, gt_cls, gt, opt.match_iou)
        matched += ious
        tp += len(ious)
        fp += fp_i
        fn += fn_i
        a, b, c = pixel_stats(pred_cls, pred, gt_cls, gt, classes, h, w)
        px_iou += a
        px_precision += b
        px_recall += c
        if i % 50 == 0:
            print(f'{i}/{len(files)}')

    results = {
        'images': len(files),
        'conf_thres': opt.conf_thres,
        'pixel': {'iou': mean(px_iou), 'precision': mean(px_precision), 'recall': mean(px_recall)},
        'instance': {
            'mean_iou_matched': mean(matched),
            'precision': tp / (tp + fp) if tp + fp else float('nan'),
            'recall': tp / (tp + fn) if tp + fn else float('nan'),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'match_iou': opt.match_iou}}
    print(json.dumps(results, indent=2))
    if opt.save_json:
        Path(opt.save_json).write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-nms', type=float, default=0.45)
    parser.add_argument('--match-iou', type=float, default=0.5)
    parser.add_argument('--device', default='')
    parser.add_argument('--save-json', default='')
    main(parser.parse_args())
