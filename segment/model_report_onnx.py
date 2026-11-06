"""Same report as segment/model_report.py, but the model runs in ONNX Runtime. Columns end in "ONNX".

Settings come from Config/main_config.yaml (sections model, dataset, map, pixel, timing, onnx).
Run from the repo root after training:
    python segment/model_report_onnx.py

If onnx.weights is empty, best.pt (model.weights) is exported to ONNX next to it first, and the export
time is reported. Needs the `onnx` package and `onnxruntime-gpu` (or `onnxruntime` for CPU). Nothing is
installed automatically.
"""
import os
import sys
import time

# Never let YOLOv5 pip-install missing packages into the shared environment. Must be set before utils.general loads.
os.environ.setdefault('YOLOv5_AUTOINSTALL', 'False')

from loguru import logger
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from segment.model_report import main as report_main, prepare_run  # noqa: E402


def check_packages(need_export):
    """Stop with a clear message instead of letting YOLOv5 try to install packages."""
    missing = []
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        missing.append('onnxruntime-gpu (or onnxruntime for CPU)')
    if need_export:
        try:
            import onnx  # noqa: F401
        except ImportError:
            missing.append('onnx')
    if missing:
        raise SystemExit(f'Missing packages: {", ".join(missing)}. The environment is shared, so install them only '
                         f'after agreeing on versions.')


def export_onnx(config):
    """Export model.weights to ONNX with export.py. Returns the .onnx path and the export time in seconds."""
    from export import run as export_run

    oc = config.onnx
    t0 = time.perf_counter()
    files = export_run(
        weights=config.model.weights,
        imgsz=(config.model.imgsz, config.model.imgsz),
        device=config.model.device,
        include=('onnx',),
        half=oc.half,
        dynamic=False,
        simplify=oc.simplify,
        opset=oc.opset)
    seconds = time.perf_counter() - t0
    onnx_files = [f for f in files if f.endswith('.onnx')]
    if not onnx_files:
        raise SystemExit(f'ONNX export failed for {config.model.weights}')
    return onnx_files[0], seconds


def check_providers(onnx_path, config):
    """Confirm ONNX Runtime really uses CUDA when a GPU is requested. It falls back to CPU silently otherwise."""
    import onnxruntime

    gpu = str(config.model.device).lower() != 'cpu'
    wanted = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
    active = onnxruntime.InferenceSession(onnx_path, providers=wanted).get_providers()
    logger.info(f'onnxruntime {onnxruntime.__version__}  available: {onnxruntime.get_available_providers()}  '
                f'active: {active}')
    if gpu and 'CUDAExecutionProvider' not in active and config.onnx.require_cuda:
        raise SystemExit('GPU requested but ONNX Runtime is not using CUDAExecutionProvider. Check that '
                         'onnxruntime-gpu matches the installed CUDA/cuDNN, or set onnx.require_cuda: false '
                         'to time on CPU.')
    return active


def main(config):
    oc = config.onnx
    onnx_path = oc.weights
    check_packages(need_export=not onnx_path)

    extra = {}
    if onnx_path:
        if not os.path.exists(onnx_path):
            raise SystemExit(f'onnx.weights not found: {onnx_path}')
        logger.info(f'Using existing ONNX model {onnx_path}')
    else:
        logger.info(f'Exporting {config.model.weights} to ONNX (opset {oc.opset}, half {oc.half})')
        onnx_path, seconds = export_onnx(config)
        extra['ONNX export time (s)'] = seconds
        logger.info(f'Exported {onnx_path} in {seconds:.1f} s')
    extra['ONNX file size (MB)'] = os.path.getsize(onnx_path) / 1E6
    extra['ONNX Runtime providers'] = check_providers(onnx_path, config)

    # The ONNX input precision is fixed at export time, so run with the same precision
    return report_main(config, weights=onnx_path, label='ONNX', half=oc.half, extra=extra)


if __name__ == '__main__':
    # Load configuration
    config = OmegaConf.load('Config/main_config.yaml')
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(config.environment.cuda_launch_blocking)

    prepare_run(config, suffix='_onnx')
    main(config)
