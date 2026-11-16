#!/bin/zsh
# Start the solar-flare analysis console inside the FermiSunPy conda environment.
# Usage: ./gui/start_gui.sh [port]   (default 8766), then open http://localhost:8766
PORT=${1:-8766}
CONDA_BASE=${CONDA_BASE:-$HOME/miniforge3}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate FermiSunPy || { echo "Could not activate the FermiSunPy conda environment"; exit 1; }
cd "$(dirname "$0")/.."
exec python gui/server.py --port "$PORT"
