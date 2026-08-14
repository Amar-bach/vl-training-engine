"""Stateless sandboxed Python executor.

Adapted from Molt's `examples/python/tools/python_executor.py` (Apache-2.0, NVIDIA),
with three deliberate changes for our setting:

  1. **Structured output.** Molt returns a single string. We return stdout/stderr *and*
     any images the snippet produced via `show()`, because a crop-and-re-measure loop
     needs to hand pictures back to the model.

  2. **Stateless, not a Jupyter kernel.** DeepEyesV2 keeps a persistent kernel per
     session; their `session_id` is a class attribute, so concurrent rollouts share one
     namespace and overwrite each other's variables (`deepeyesv2.py:76`). We give every
     call a fresh process seeded from an explicit preamble. Continuity across turns is
     the model's job (re-state what it needs), which costs a little compute and buys
     exact reproducibility of a rollout from its transcript.

  3. **No network.** Molt's docstring notes chroot is not feasible from userland and
     relies on timeout + memory cap. We additionally disable the socket module in the
     preamble. This is a guard rail against accidental egress, not a security boundary:
     as upstream says, do NOT expose this to untrusted input over the internet.

Isolation: `python3 -I` (no user site, no env), throwaway cwd, RLIMIT_AS soft 1 GiB /
hard x1.5, RLIMIT_CPU, hard wall-clock timeout, truncated output. Never raises.
"""

from __future__ import annotations

import base64
import glob
import io
import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# The interpreter the snippet runs under. Defaults to the harness's own interpreter so
# the sandbox sees the same numpy/PIL. `python3` from PATH would be the *system* python
# under conda and would have neither -- a mistake that shows up as every call failing
# with ModuleNotFoundError inside the preamble, i.e. as a dead tool.
DEFAULT_PYTHON = os.environ.get('SURDS_SANDBOX_PYTHON') or sys.executable or 'python3'

DEFAULT_TIMEOUT_SECONDS = float(os.environ.get('SURDS_SANDBOX_TIMEOUT', '10'))
# 4 GiB, not Molt's 1 GiB. RLIMIT_AS caps *virtual address space*, not resident memory,
# and numpy's OpenBLAS reserves per-core thread arenas at import. On this cluster
# (255 cores) a 1 GiB cap makes `import numpy` HANG rather than fail -- measured, see
# `_THREAD_ENV` below. Molt never hit this because their geo3k preamble only imports
# `math`; ours imports numpy, which is the point of the tool.
DEFAULT_MEM_LIMIT_BYTES = int(os.environ.get('SURDS_SANDBOX_MEM_BYTES', str(4 * 1024 * 1024 * 1024)))

# Pin BLAS to one thread in the child. Two independent reasons:
#   (a) it is the actual fix for the hang above -- with threads pinned, even a 1 GiB
#       RLIMIT_AS imports numpy in 0.13 s;
#   (b) a tool call is a few hundred numbers of geometry. Letting 16 concurrent rollouts
#       each spawn 255 BLAS threads would thrash the node the trainer is running on.
_THREAD_ENV = {
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
}
DEFAULT_OUTPUT_CHARS = int(os.environ.get('SURDS_SANDBOX_OUTPUT_CHARS', '2048'))
DEFAULT_MAX_IMAGES = int(os.environ.get('SURDS_SANDBOX_MAX_IMAGES', '2'))

# Written into the child's cwd; the parent collects them after the run.
_IMG_PREFIX = '_show_'

_PREAMBLE = '''\
import sys as _sys, math, io as _io
try:
    import socket as _socket
    def _no_net(*a, **k):
        raise OSError("network access is disabled in this sandbox")
    _socket.socket = _no_net
    _socket.create_connection = _no_net
except Exception:
    pass

import numpy as np
from PIL import Image

_SHOW_COUNT = [0]

def show(obj):
    """Return an image to the model. Accepts a PIL Image or a HxW / HxWx3 array."""
    global _SHOW_COUNT
    if _SHOW_COUNT[0] >= {max_images}:
        print("[show] image limit ({max_images}) reached; ignoring extra image")
        return
    im = obj
    if not isinstance(im, Image.Image):
        arr = np.asarray(obj)
        if arr.dtype != np.uint8:
            finite = arr[np.isfinite(arr)] if arr.size else arr
            lo = float(finite.min()) if finite.size else 0.0
            hi = float(finite.max()) if finite.size else 1.0
            rng = (hi - lo) or 1.0
            arr = ((arr - lo) / rng * 255.0).clip(0, 255).astype("uint8")
        im = Image.fromarray(arr)
    im.convert("RGB").save("{prefix}%02d.png" % _SHOW_COUNT[0])
    _SHOW_COUNT[0] += 1

{context}
'''


@dataclass
class ExecResult:
    stdout: str = ''
    stderr: str = ''
    images: List[Any] = field(default_factory=list)   # PIL.Image.Image
    ok: bool = True
    returncode: Optional[int] = None
    error: str = ''
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        """Observation text in DeepEyesV2's validated layout (prompt.py:1-13)."""
        parts = []
        if self.error:
            parts.append(self.error)
        parts.append(f'stdout:\n```\n{self.stdout}\n```')
        if self.stderr:
            parts.append(f'stderr:\n```\n{self.stderr}\n```')
        if self.images:
            parts.append('Images:\n' + '<image>' * len(self.images))
        return '\n\n'.join(parts)


def _set_limits(mem_bytes: int, cpu_seconds: int):
    def _apply():
        for res, limit in ((resource.RLIMIT_AS, (mem_bytes, int(mem_bytes * 1.5))),
                           (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))):
            try:
                resource.setrlimit(res, limit)
            except (ValueError, OSError):
                pass  # some environments forbid setrlimit; timeout still applies
    return _apply


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    head = text[:cap - 64]
    return f'{head}\n... [{len(text) - len(head)} more chars truncated]'


def build_context_preamble(image_path: Optional[str] = None,
                           intrinsics: Optional[Any] = None) -> str:
    """Seed the namespace with the sample's frame and camera geometry.

    `img` is the NATIVE 1600x900 image, so sandbox coordinates == native coordinates.
    This is a choice, and it is documented in `frames.py` so it never becomes an open
    question. Camera intrinsics are the SURDS-native substitute for DeepEyesV2's web
    search: the information the pixels alone cannot supply.

    Never expose nuScenes 3-D box annotations here -- that is the label.
    """
    lines = []
    if image_path:
        lines.append(f'img = Image.open({image_path!r}).convert("RGB")')
        lines.append('W, H = img.size')
    if intrinsics is not None:
        lines.append(f'K = np.array({list(map(list, intrinsics))!r}, dtype=float)')
    return '\n'.join(lines)


def run_python(code: str,
               *,
               context_preamble: str = '',
               python: str = DEFAULT_PYTHON,
               timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
               mem_bytes: int = DEFAULT_MEM_LIMIT_BYTES,
               output_chars: int = DEFAULT_OUTPUT_CHARS,
               max_images: int = DEFAULT_MAX_IMAGES) -> ExecResult:
    """Execute `code` in a fresh sandboxed interpreter. Never raises.

    Note we ship the model's code UNMODIFIED. DeepEyesV2 re-indents it with a
    `line.endswith(':')` heuristic plus aggressive autopep8 (`deepeyesv2.py:53-70`),
    which corrupts multi-line and triple-quoted strings. If the model's code is broken,
    the traceback is the observation and that is a legitimate training signal.
    """
    if not isinstance(code, str) or not code.strip():
        return ExecResult(ok=False, error='empty code argument.')

    preamble = _PREAMBLE.format(max_images=max_images, prefix=_IMG_PREFIX,
                                context=context_preamble)
    script = f'{preamble}\n{code}\n'

    try:
        with tempfile.TemporaryDirectory(prefix='surds_sandbox_') as workdir:
            proc = subprocess.run(
                [python, '-I', '-c', script],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=_set_limits(mem_bytes, int(timeout_seconds) + 1),
                cwd=workdir,
                env={**os.environ, **_THREAD_ENV},
            )
            images = _collect_images(workdir, max_images)
    except subprocess.TimeoutExpired:
        return ExecResult(ok=False, error=f'execution timed out after {timeout_seconds:.1f}s.')
    except Exception as exc:  # defensive: a launch failure must not kill the rollout
        return ExecResult(ok=False,
                          error=f'failed to launch interpreter ({type(exc).__name__}: {exc}).')

    stdout = _truncate(proc.stdout or '', output_chars)
    stderr = _truncate(proc.stderr or '', output_chars)
    ok = proc.returncode == 0
    return ExecResult(
        stdout=stdout if stdout else ('(no output)' if ok else ''),
        stderr=stderr,
        images=images,
        ok=ok,
        returncode=proc.returncode,
        error='' if ok else f'Exit code {proc.returncode}.',
        meta={'n_images': len(images)},
    )


def _collect_images(workdir: str, max_images: int) -> List[Any]:
    """Read back whatever `show()` wrote, before the temp dir disappears."""
    from PIL import Image
    out = []
    for path in sorted(glob.glob(os.path.join(workdir, f'{_IMG_PREFIX}*.png')))[:max_images]:
        try:
            with open(path, 'rb') as fh:
                out.append(Image.open(io.BytesIO(fh.read())).convert('RGB'))
        except Exception:
            continue  # a corrupt image is not worth failing the call over
    return out


def encode_png_b64(image) -> str:
    """Helper for the HTTP transport (sandbox/server.py)."""
    buf = io.BytesIO()
    image.convert('RGB').save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')
