from __future__ import annotations

import os


_ORIGINAL_MKDIR = os.mkdir


def _lektorai_mkdir(path, mode=0o777, *args, **kwargs):
    if os.name == "nt" and mode == 0o700:
        mode = 0o755
    return _ORIGINAL_MKDIR(path, mode, *args, **kwargs)


os.mkdir = _lektorai_mkdir
