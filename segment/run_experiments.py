"""Train several YOLOv5 segmentation models one after another, then run model_report.py on each best.pt.

Settings come from the `experiments` section of Config/main_config.yaml (report settings from the other sections).
Run from the repo root, for example in the background so it survives closing the terminal:
    nohup python segment/run_experiments.py > bin/experiments_nohup.log 2>&1 &

Per experiment:
    1. Skip training if <project>/<name>/weights/best.pt is from a finished training and experiments.skip_done is
       true. Finished = train.py stripped the optimizer at the end; a run that died keeps it in best.pt.
    2. An unfinished run folder with the same name is renamed to <name>_incomplete_<timestamp> (never deleted).
       This includes a run whose last attempt failed, so its partial best.pt is never reported or skipped.
    3. Train with segment/train.py. On CUDA out of memory, retry with each batch size in experiments.oom_batch_sizes.
    4. Run the Torch report (and the ONNX report if experiments.report_onnx) on best.pt, each in its own process,
       so no GPU memory is left reserved when the next training starts.
A failed experiment is logged and the next one starts.

Outputs in bin/Run_<timestamp>_experiments/:
    config_used.yaml, runner.log, train/<name>_bs<batch>.log, experiments.csv (one row per experiment, everything)
    experiments_sheet.csv / .tsv: Model + results-sheet columns (val set accuracy, then Pre/Model/Post ms)
Each report also appends to report.summary_csv and has its own bin/Run_<timestamp>_<name>_torch folder.
"""
from datetime import datetime
import json
import multiprocessing as mp
import os
import queue as queue_module
import socket
import subprocess
import sys
import time

# Never let YOLOv5 pip-install missing packages into the shared environment. Must be set before utils.general loads.
os.environ.setdefault('YOLOv5_AUTOINSTALL', 'False')

from loguru import logger
from omegaconf import OmegaConf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from segment import model_report  # noqa: E402

OOM_MARKERS = ('CUDA out of memory', 'OutOfMemoryError', 'CUBLAS_STATUS_ALLOC_FAILED')


def train_command(config, run, batch_size):
    """segment/train.py command for one experiment."""
    ex = config.experiments
    cmd = [sys.executable, os.path.join(ROOT, 'segment', 'train.py'),
           '--weights', run.weights,
           '--data', ex.data_yaml or config.dataset.data_yaml,
           '--imgsz', str(run.get('imgsz', 640)),
           '--epochs', str(run.get('epochs', ex.epochs)),
           '--batch-size', str(batch_size),
           '--device', str(ex.device),
           '--workers', str(ex.workers),
           '--project', ex.project,
           '--name', run.name]
    if run.get('cfg'):  # model yaml, e.g. models/segment/yolov5n-seg-proto2x.yaml (weights still load where shapes match)
        cmd += ['--cfg', run.cfg]
    if run.get('hyp'):
        cmd += ['--hyp', run.hyp]
    if run.get('extra_args'):
        cmd += str(run.extra_args).split()
    return cmd


def run_logged(cmd, log_path):
    """Run a command, stream its output to the console and a log file. Return (exit code, saw out-of-memory)."""
    env = dict(os.environ, YOLOv5_AUTOINSTALL='False')
    oom = False
    with open(log_path, 'w') as log:
        log.write(' '.join(cmd) + '\n\n')
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            oom = oom or any(m in line for m in OOM_MARKERS)
        proc.wait()
    return proc.returncode, oom


def move_incomplete(run_dir):
    """Rename an unfinished run folder so a new attempt starts clean (train.py appends to results.csv otherwise)."""
    if os.path.isdir(run_dir):
        new = f"{run_dir}_incomplete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(run_dir, new)
        logger.warning(f'Moved unfinished run folder {run_dir} -> {new}')


def training_finished(best):
    """True if best.pt comes from a training that ran to the end: train.py strips the optimizer (sets it to None)
    from best.pt only after the last epoch. A best.pt saved during training still holds the optimizer state."""
    if not os.path.exists(best):
        return False
    try:
        return torch.load(best, map_location='cpu', weights_only=False).get('optimizer') is None
    except Exception as e:  # unreadable checkpoint (e.g. killed while saving)
        logger.warning(f'Cannot read {best}: {e!r}')
        return False


def write_runner_info(run_dir, run, t_start, total_minutes, batch_size, note):
    """<run_dir>/runner_info.json: training time and machine, read by segment/experiment_registry.py."""
    end = time.time()
    info = {'name': run.name,
            'start': datetime.fromtimestamp(t_start).isoformat(timespec='seconds'),
            'end': datetime.fromtimestamp(end).isoformat(timespec='seconds'),
            'train_hours': round((end - t_start) / 3600, 3),  # successful attempt only
            'total_minutes_incl_retries': round(total_minutes, 1),
            'batch_size': batch_size,
            'note': note.strip(' ()'),
            'host': socket.gethostname(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}
    with open(os.path.join(run_dir, 'runner_info.json'), 'w') as f:
        json.dump(info, f, indent=2)


def train(config, run, log_dir):
    """Train one experiment with out-of-memory retries. Return (status, batch size used, minutes)."""
    ex = config.experiments
    run_dir = os.path.join(ROOT, ex.project, run.name)
    best = os.path.join(run_dir, 'weights', 'best.pt')
    if ex.skip_done and training_finished(best):
        logger.info(f'{run.name}: finished best.pt exists, training skipped')
        return 'skipped (already trained)', '', 0.0
    if os.path.exists(best):
        logger.warning(f'{run.name}: best.pt is from an unfinished training, training again')

    t0 = time.time()
    # per-run oom_batch_sizes overrides the global list; [] = no retry (a batch size experiment must not silently
    # fall back to a smaller batch)
    for batch_size in [run.get('batch_size', ex.batch_size), *run.get('oom_batch_sizes', ex.oom_batch_sizes)]:
        move_incomplete(run_dir)
        cmd = train_command(config, run, batch_size)
        logger.info(f"{run.name}: training, batch {batch_size}\n  {' '.join(cmd)}")
        t_attempt = time.time()
        code, oom = run_logged(cmd, os.path.join(log_dir, f'{run.name}_bs{batch_size}.log'))
        minutes = (time.time() - t0) / 60
        if code == 0 and training_finished(best):
            note = '' if batch_size == run.get('batch_size', ex.batch_size) else f' (out of memory, retried with batch {batch_size})'
            logger.info(f'{run.name}: training done in {minutes:.0f} min{note}')
            write_runner_info(run_dir, run, t_attempt, minutes, batch_size, note)
            return f'trained{note}', batch_size, minutes
        if oom:
            logger.warning(f'{run.name}: CUDA out of memory with batch {batch_size}')
            continue
        logger.error(f'{run.name}: training failed (exit code {code}), see log in {log_dir}')
        move_incomplete(run_dir)  # a partial best.pt must not be reported or skipped as done later
        return f'failed (exit code {code})', batch_size, minutes
    move_incomplete(run_dir)
    return 'failed (out of memory at every batch size)', '', (time.time() - t0) / 60


def report_worker(cfg_dict, backend, queue):
    """Child process: run one report and send (result, output folder or error) back. Its GPU memory is fully
    released when the process exits, so the next training starts with a clean GPU."""
    result, out = None, 'failed: no result'
    try:
        cfg = OmegaConf.create(cfg_dict)
        os.environ['CUDA_LAUNCH_BLOCKING'] = str(cfg.environment.cuda_launch_blocking)
        if backend == 'onnx':
            from segment import model_report_onnx
            model_report.prepare_run(cfg, suffix='_onnx')
            result = model_report_onnx.main(cfg)
        else:
            model_report.prepare_run(cfg, suffix='_torch')
            result = model_report.main(cfg)
        out = cfg.dataset.save_path
    except (Exception, SystemExit) as e:  # one failed report must not stop the queue
        out = f'failed: {e!r}'
    finally:
        queue.put((result, out))


def report(config, run, backend):
    """Run model_report (Torch or ONNX) on the experiment's best.pt in a separate process.
    Return (report dict or None, output folder or error)."""
    ex = config.experiments
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.model_name = run.name
    cfg.model.weights = os.path.join(ROOT, ex.project, run.name, 'weights', 'best.pt')
    cfg.model.imgsz = run.get('imgsz', 640)
    cfg.dataset.data_yaml = ex.data_yaml or config.dataset.data_yaml
    if backend == 'onnx':
        cfg.onnx.weights = ''  # export this best.pt

    ctx = mp.get_context('spawn')  # fresh process: no CUDA state shared with the runner or the next training
    queue = ctx.Queue()
    proc = ctx.Process(target=report_worker, args=(OmegaConf.to_container(cfg, resolve=True), backend, queue))
    proc.start()
    result, out = None, None
    while out is None:
        try:
            result, out = queue.get(timeout=10)
        except queue_module.Empty:
            if not proc.is_alive():  # crashed without sending a result (e.g. killed, segfault)
                out = f'failed: report process exited with code {proc.exitcode}'
    proc.join()
    if result is None:
        logger.error(f'{run.name}: {backend} report {out}')
    return result, out


def main(config):
    ex = config.experiments
    if not torch.cuda.is_available() or str(ex.device).lower() == 'cpu':
        raise SystemExit('Experiments must run on GPU, but CUDA is not available or experiments.device is cpu.')

    rows, sheet_rows = [], []
    for k, run in enumerate(ex.runs, 1):
        logger.info(f'===== Experiment {k}/{len(ex.runs)}: {run.name} =====')
        status, batch_used, minutes = train(config, run, config.dataset.train_output)
        row = {'Experiment': run.name, 'Weights': run.weights, 'Imgsz': run.get('imgsz', 640),
               'Hyp': run.get('hyp', '') or 'default', 'Extra args': run.get('extra_args', ''),
               'Status': status, 'Batch size': batch_used, 'Train minutes': round(minutes, 1)}
        if training_finished(os.path.join(ROOT, ex.project, run.name, 'weights', 'best.pt')):
            for backend in (['torch'] if ex.report_torch else []) + (['onnx'] if ex.report_onnx else []):
                result, out = report(config, run, backend)
                row[f'{backend} report'] = out
                if result:
                    row.update(result['report'])
                    sheet_rows.append({'Model': run.name, **result['sheet']})
        rows.append(row)
        model_report.write_csv(os.path.join(config.dataset.save_path, 'experiments.csv'), rows)
        if sheet_rows:  # results-sheet columns only, one row per model (and backend), updated after each run
            for name, delim in (('experiments_sheet.csv', ','), ('experiments_sheet.tsv', '\t')):
                model_report.write_csv(os.path.join(config.dataset.save_path, name), sheet_rows, delimiter=delim)
        logger.info(f'{run.name}: {status}')

    logger.info('===== All experiments finished =====')
    for r in rows:
        logger.info(f"  {r['Experiment']:<45} {r['Status']}")
    return rows


if __name__ == '__main__':
    # Load configuration
    config = OmegaConf.load('Config/main_config.yaml')
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(config.environment.cuda_launch_blocking)

    # Generate timestamp and run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = os.path.join('bin', f'Run_{timestamp}_experiments')
    os.makedirs(os.path.join(checkpoints_dir, "train"), exist_ok=True)

    # Update config paths
    config.dataset.save_path = checkpoints_dir
    config.dataset.train_output = os.path.join(checkpoints_dir, "train")  # training logs per experiment

    # Save the config used for this run
    OmegaConf.save(config, os.path.join(checkpoints_dir, "config_used.yaml"))

    model_report.prepare_logger(checkpoints_dir, 'runner.log')
    logger.info(f'Experiments output directory: {checkpoints_dir}')
    main(config)
