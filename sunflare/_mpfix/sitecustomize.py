"""Make GtBurst's parallel exposure maps start their workers with 'fork' (only for gtapps_mp/gtexpmap_mp.py).

sunflare puts this directory first on PYTHONPATH for the processes it launches, so Python runs this file at start-up.

Why: GtBurst computes exposure maps with gtapps_mp/gtexpmap_mp.py, which splits the sky into tiles and runs gtexpmap
in a multiprocessing.Pool. On macOS the default start method is 'spawn', so every worker re-imports gt_apps, which
re-reads ./gtexpmap.par while gtexpmap executables in other workers are rewriting that same file. Workers that hit
that moment die at start-up with "FileNotFoundError: ... './gtexpmap.par'" (the pool replaces them, so the maps are
complete, but the logs fill with tracebacks). With 'fork', the workers inherit the parameters already loaded by the
parent and never read the file. GtBurst was written for Linux, where 'fork' is the default.
"""
import os
import sys

if sys.argv and os.path.basename(sys.argv[0]) == 'gtexpmap_mp.py':
    try:
        import multiprocessing
        multiprocessing.set_start_method('fork', force=True)
    except Exception:
        pass
    try:   # forked workers exit without flushing: line-buffer stdout so their progress messages reach the log
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
