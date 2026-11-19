#!/usr/bin/env bash
# Wait, then start the experiment queue (segment/run_experiments.py, settings in Config/main_config.yaml).
#
# Usage, from anywhere (the script changes to the repository root itself):
#   nohup ./run_experiments_delayed.sh > delayed_start.log 2>&1 &     # survives closing the terminal / SSH
#   ./run_experiments_delayed.sh 3                                      # optional: delay in hours (default 2)
#
# Python: the active environment's `python`, or set PYTHON, e.g.
#   PYTHON=/home/aiviz05/venv_studio/generic_env/bin/python3.10 nohup ./run_experiments_delayed.sh > delayed_start.log 2>&1 &
#
# Cancel before the start: kill the script's process (`pkill -f run_experiments_delayed.sh`).

set -euo pipefail

DELAY_HOURS="${1:-2}"
PYTHON="${PYTHON:-python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

delay_seconds=$(awk -v h="$DELAY_HOURS" 'BEGIN { printf "%d", h * 3600 }')
start_at=$(date -d "+${delay_seconds} seconds" '+%Y-%m-%d %H:%M:%S')

cd "$REPO_DIR"
"$PYTHON" -c "import torch" || { echo "ERROR: '$PYTHON' has no torch; activate the training environment or set PYTHON"; exit 1; }

echo "$(date '+%Y-%m-%d %H:%M:%S') waiting ${DELAY_HOURS} h; experiments start at ${start_at} (PID $$)"
sleep "$delay_seconds"

echo "$(date '+%Y-%m-%d %H:%M:%S') starting: $PYTHON segment/run_experiments.py"
code=0
"$PYTHON" segment/run_experiments.py || code=$?
echo "$(date '+%Y-%m-%d %H:%M:%S') run_experiments.py finished with exit code ${code}"
exit "$code"
