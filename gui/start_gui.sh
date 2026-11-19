#!/bin/zsh
# The solar-flare console moved to the fermiconsole repository (sun flavor). This wrapper starts it.
#   ./gui/start_gui.sh [port] [--port PORT] [--host HOST] [-h]
FC=${FERMICONSOLE_PATH:-$HOME/gitrepository/fermiconsole}
if [[ ! -x "$FC/start_console.sh" ]]; then
  echo "fermiconsole not found in $FC: clone https://github.com/omodei/fermiconsole there or set FERMICONSOLE_PATH"
  exit 1
fi
exec "$FC/start_console.sh" sun "$@"
