"""Build a YOLO dataset that keeps only some classes of an existing YOLO segmentation dataset.

Settings: class_subset in Config/main_config.yaml. Run from the repo root:  python segment/make_class_subset.py

Images are hard links to the source images (no extra disk space, source and target must be on the same drive);
label files are rewritten with only the kept classes, renumbered 0, 1, ... in the order of class_subset.keep.
Images whose labels lose every object stay in the dataset as background images.
Example: 2-class data (0 = product, 1 = cavity) -> product only, to compare a 1-class model on the same images.
"""
from datetime import datetime
import os
import shutil

from omegaconf import OmegaConf
from loguru import logger


def prepare_logger(log_dir, log_filename):
    """Configure loguru logger with file output"""
    log_path = os.path.join(log_dir, log_filename)
    logger.remove()  # Remove default handler
    logger.add(log_path, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
    logger.add(lambda msg: print(msg, end=""))  # Keep console output


def build_split(cs, split, remap):
    src_img, src_lab = os.path.join(cs.source, split, 'images'), os.path.join(cs.source, split, 'labels')
    dst_img, dst_lab = os.path.join(cs.target, split, 'images'), os.path.join(cs.target, split, 'labels')
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lab, exist_ok=True)
    n_img = n_obj = n_empty = 0
    for f in sorted(os.listdir(src_img)):
        dst = os.path.join(dst_img, f)
        if not os.path.exists(dst):
            os.link(os.path.join(src_img, f), dst)
        n_img += 1
        lab = os.path.splitext(f)[0] + '.txt'
        lines = []
        if os.path.exists(os.path.join(src_lab, lab)):
            with open(os.path.join(src_lab, lab)) as fh:
                for line in fh:
                    p = line.split()
                    if p and int(p[0]) in remap:
                        lines.append(' '.join([str(remap[int(p[0])])] + p[1:]) + '\n')
        with open(os.path.join(dst_lab, lab), 'w') as fh:
            fh.writelines(lines)
        n_obj += len(lines)
        n_empty += not lines
    logger.info(f'{split}: {n_img} images, {n_obj} objects kept, {n_empty} images without objects')


def main(config):
    cs = config.class_subset
    remap = {int(c): i for i, c in enumerate(cs.keep)}
    if os.path.exists(os.path.join(cs.target, 'train')) and not cs.overwrite:
        raise SystemExit(f'{cs.target} already exists; set class_subset.overwrite: true to rebuild it')
    for split in cs.splits:
        if cs.overwrite:
            shutil.rmtree(os.path.join(cs.target, split), ignore_errors=True)
        build_split(cs, split, remap)
    yaml_path = os.path.join(cs.target, os.path.basename(cs.yaml_name))
    with open(yaml_path, 'w') as fh:
        fh.write(f'# Made by segment/make_class_subset.py on {datetime.now():%Y-%m-%d} from {cs.source}\n'
                 f'# (source classes kept: {list(cs.keep)}). Images are hard links to the source images.\n')
        OmegaConf.save(OmegaConf.create({'path': cs.target, 'train': 'train/images', 'val': 'val/images',
                                         'names': {i: str(n) for i, n in enumerate(cs.names)}}), fh)
    logger.info(f'Dataset yaml: {yaml_path}')
    return yaml_path


if __name__ == '__main__':
    # Load configuration
    config = OmegaConf.load('Config/main_config.yaml')

    # Generate timestamp and run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join('bin', f'Run_{timestamp}_class_subset')
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(run_dir, 'config_used.yaml'))
    prepare_logger(run_dir, 'class_subset.log')

    main(config)
