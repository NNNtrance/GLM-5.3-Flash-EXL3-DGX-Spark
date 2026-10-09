# Stack-dump hook for the hang protocol: SIGUSR1 -> faulthandler (all threads; stderr =
# the container log). The engine code does not change; only a C-level signal handler is
# installed (no cost on the hot path). The image's own sitecustomize
# (/usr/lib/python3.12/sitecustomize.py, the apport hook) is kept exactly as it is. This
# file comes in through PYTHONPATH=/opt/harem-yigin.
# Lesson: in a worker process SIGUSR1 is overridden by another handler, and the main thread
# waits in C (cudaStreamSynchronize), so no stack gets dumped. Therefore, when
# HAREM_YIGIN_SN>0, every process dumps the stacks of ALL threads ONCE, N seconds after its
# start (faulthandler's own watchdog thread; it works even when the main thread is stuck in
# C; the process does not die, nothing changes). Keep HAREM_YIGIN_SN=0 on a live engine: a
# timed dump has crashed a worker with SIGSEGV.
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()
try:
    import faulthandler as _harem_fh
    import signal as _harem_sg
    _harem_fh.register(_harem_sg.SIGUSR1, all_threads=True, chain=False)
except Exception:  # noqa: BLE001 -- the stack hook never breaks a boot
    pass
try:
    import os as _harem_os
    _harem_sn = int(_harem_os.environ.get("HAREM_YIGIN_SN", "0") or 0)
    if _harem_sn > 0:
        import faulthandler as _harem_fh2
        import sys as _harem_sys
        _harem_dz = _harem_os.environ.get("HAREM_YIGIN_DIZIN", "")
        _harem_tekrar = _harem_os.environ.get("HAREM_YIGIN_TEKRAR", "") == "1"
        _harem_dosya = _harem_sys.stderr
        if _harem_dz and _harem_os.path.isdir(_harem_dz):
            # one file per process (the diagnostic run): every thread, workers included
            _harem_ad = _harem_os.path.basename(_harem_sys.argv[0] or "py")[:24]
            _harem_dosya = open(_harem_os.path.join(_harem_dz, f"yigin-{_harem_os.uname().nodename}-{_harem_os.getpid()}-{_harem_ad}.txt"), "a", buffering=1)
        _harem_fh2.dump_traceback_later(_harem_sn, repeat=_harem_tekrar, file=_harem_dosya, exit=False)
except Exception:  # noqa: BLE001
    pass
