# Plan: train on tiles, test on the complete image

Status: **planned, not implemented.** Start when the V18 images are available on the laptop
(`/media/mostafahaggag/D/yolov5`), so step 0 can run here.

## Why

The mask mAP50-95 of all trained runs stops at about 0.65-0.69 after epoch 15-20, while box mAP50-95 keeps
rising to 0.84-0.87. A bigger model (yolov5s) and no mosaic did not change it. The limit is the mask resolution:

- The images are wide strips (about 3:1). At imgsz 640 the image is only about 210 px tall.
- Pills are 0.025-0.06 of the image width, so 16-38 px at 640.
- YOLOv5 builds masks on a stride-4 prototype grid (160x160 at 640), so a pill is only 4-10 mask cells wide.

Tiling makes the pills bigger in pixels during training, without the memory cost of a huge imgsz.

## Goal

- **Training:** square tiles cut from each image, at high resolution, so pills are big in pixels.
- **Validation, report and inspection:** the complete image in one pass, resized with the **same scale factor**
  as the tiles. Pills then have the same pixel size the model learned. No tiling or merging at inspection.

Rule: the pill size in pixels at inspection must match the pill size in training.

```
scale factor     = train imgsz / tile size            (e.g. 640 / 1000 = 0.64)
report imgsz     = image width x scale factor          (e.g. 3000 x 0.64 = 1920, rounded to a multiple of 32)
```

## Step 0: image statistics

New script `segment/tile_dataset.py`. Settings come from a new `tiling` section in `Config/main_config.yaml`
(OmegaConf, no argparse, output in `bin/Run_<timestamp>_tiling/`).

With `tiling.mode: stats` it only reads the dataset and logs:

- Image width and height: minimum, median, maximum, and the number of different sizes.
- Pill width and height in pixels.
- Suggested tile size, overlap, training imgsz and report imgsz, plus the expected compute cost compared with today.

## Step 1: build the tiled dataset (`tiling.mode: tile`)

- **Tiles:** square, side = image height by default (`tiling.tile_size`, 0 = image height).
  Overlap at least the widest pill, 20% by default (`tiling.overlap`). The last tile is aligned to the right edge,
  so there are no partial tiles.
  Example: 3000 px wide, tile 1000, overlap 200, step 800: tiles start at x = 0, 800, 1600, 2000.
- **Labels:** each pill polygon is converted to pixels and clipped to the tile rectangle
  (Sutherland-Hodgman in numpy, no new package in the shared environment).
  - Visible part at least 25% of the pill area (`tiling.min_visible`): keep the clipped polygon.
    Do not drop it, or the model learns that half-pills are background.
  - Smaller sliver: drop it.
  - Polygons are normalized to the tile again and written as YOLO-seg labels.
- **Empty tiles:** keep a fraction, about 10% (`tiling.keep_empty`).
- **Split by original image:** all tiles of one image go to the same split, so train and val never share an image.
- **Output:** a new local folder and a new data `.yaml`. The original dataset is only read. Nothing is written to
  the NAS.
- **Check images:** `tiling.num_check` tiles drawn with their clipped polygons, to verify the labels by eye.
- **Logs:** tiles per image, pills kept, clipped and dropped.

## Step 2: run settings in `segment/run_experiments.py`

Two new optional keys per run:

- `train_data`: the tiled data yaml, used for training only.
- `report_imgsz`: the imgsz for the report on the complete val images. Default from the formula above.

The report keeps using the **original** val set (complete images), so its numbers are directly comparable with all
other runs and with Mask R-CNN.

Example queue entry:

```yaml
- {name: v18_yolov5n_seg_tiles640, weights: yolov5n-seg.pt, imgsz: 640, hyp: data/hyps/hyp.bismart.yaml,
   train_data: <tiled data yaml>, report_imgsz: 1920}
```

## Step 3: report on complete images

- If all images have the same size: no change in the report, only `model.imgsz = report_imgsz`.
  Torch inference already uses rect letterbox (`auto=True`).
- If image sizes differ: a fixed imgsz gives a different pill scale per image. Then add `model.scale` to
  `segment/model_report.py` (fixed scale factor per image instead of a fixed long side) for timing, IoU/P/R and the
  val.py mAP dataloader. Step 0 decides whether this is needed.
- ONNX has a fixed input size, so export at `report_imgsz`.

## Step 4: tests (CPU, on the laptop)

- Clipping: polygon fully inside, cut on one edge, cut at a corner, sliver dropped, concave shape.
- Tile coordinates: rebuild complete-image masks from the tiles and compare with the original masks
  (IoU about 1 in the overlap zones).
- End-to-end on a few images: tile, load with the segmentation dataloader, forward pass and loss.

## Open questions (answer before implementing)

1. Original image resolution of the V18 images (step 0 answers this).
2. Where to write the tiled dataset on the training PC, on a local disk, not the NAS
   (for example `/home/aiviz05/datasets/BISMART_tiles`).
3. Inspection time budget in ms per image. It sets the default tile size:
   bigger tiles give a smaller report imgsz, which is faster but gives coarser masks.

## Risks

- Inspection compute grows with `report_imgsz`. 1920x640 is about 3x the pixels of a 640x640 pass,
  so model time roughly 2-3x today. Step 0 estimates this before any training.
- Pills cut at tile edges are seen partially in training. Overlap and `min_visible` handle this.
  The complete image at inspection never cuts pills.
- Training val curves are computed on tiles. Only the report on complete images counts.

## Related runs already in the queue

- `v18_yolov5n_seg_960_nomosaic`, `*_960_rect`, `*_1280_rect`: bigger imgsz without tiling.
- `*_proto2x`: masks at stride 2 (320x320 at 640) with the backbone unchanged, the cheapest option for speed.

Compare tiling against these in `experiments_sheet.tsv` (Mask mAP75, Mask mAP50-95, Torch Model/Post ms).
