"""Unit tests for the sandbox executor.

These mirror Molt's `tests/unit/test_python_executor.py` in spirit: the contract under
test is mostly about what must NOT happen (raise, hang, leak, blow up the prompt).

Run:  python -m pytest research/agentic/tests/test_sandbox.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agentic.sandbox import executor as E  # noqa: E402


def test_basic_stdout():
    r = E.run_python('print(2 + 2)')
    assert r.ok and r.stdout.strip() == '4'


def test_numpy_available():
    r = E.run_python('import numpy; print(numpy.array([1,2,3]).sum())')
    assert r.ok and r.stdout.strip() == '6'


def test_np_and_math_preloaded():
    r = E.run_python('print(int(np.sqrt(16)), int(math.floor(2.7)))')
    assert r.ok and r.stdout.strip() == '4 2'


# --------------------------------------------------------------------------------------
# Never raises -- every failure mode becomes an observation
# --------------------------------------------------------------------------------------

def test_syntax_error_is_an_observation_not_an_exception():
    # NB `this is not python` is *valid* syntax (an `is not` comparison) -- use
    # something the parser actually rejects.
    r = E.run_python('def (:')
    assert not r.ok
    assert 'SyntaxError' in r.stderr


def test_runtime_error_captured():
    r = E.run_python('1/0')
    assert not r.ok and 'ZeroDivisionError' in r.stderr


def test_timeout_does_not_raise():
    r = E.run_python('while True: pass', timeout_seconds=1.0)
    assert not r.ok and 'timed out' in r.error


def test_empty_code():
    for bad in ['', '   ', None, 123]:
        r = E.run_python(bad)
        assert not r.ok and 'empty code' in r.error


def test_memory_cap_enforced():
    r = E.run_python('x = bytearray(4 * 1024 * 1024 * 1024)',
                     mem_bytes=256 * 1024 * 1024, timeout_seconds=20)
    assert not r.ok


# --------------------------------------------------------------------------------------
# Isolation
# --------------------------------------------------------------------------------------

def test_network_is_disabled():
    r = E.run_python('import socket; socket.socket()')
    assert not r.ok and 'network access is disabled' in r.stderr


def test_cwd_is_throwaway(tmp_path):
    """A snippet that writes files must not pollute the caller's cwd."""
    before = set(os.listdir('.'))
    r = E.run_python('open("pollution.txt", "w").write("x")')
    assert r.ok
    assert set(os.listdir('.')) == before


def test_state_does_not_persist_between_calls():
    """Statelessness is the design choice that avoids DeepEyesV2's shared-kernel bug."""
    E.run_python('leaked_variable = 42')
    r = E.run_python('print(leaked_variable)')
    assert not r.ok and 'NameError' in r.stderr


def test_code_is_not_rewritten():
    """We ship the model's code unmodified; DeepEyesV2's re-indenter would corrupt this."""
    code = 's = """line1\n  line2\n    line3"""\nprint(len(s.splitlines()))'
    r = E.run_python(code)
    assert r.ok and r.stdout.strip() == '3'


# --------------------------------------------------------------------------------------
# Output discipline
# --------------------------------------------------------------------------------------

def test_output_is_truncated():
    r = E.run_python('print("x" * 100000)', output_chars=512)
    assert r.ok and len(r.stdout) <= 512 + 80
    assert 'truncated' in r.stdout


def test_show_returns_images():
    r = E.run_python('show(np.zeros((40, 60, 3)))')
    assert r.ok and len(r.images) == 1
    assert r.images[0].size == (60, 40)


def test_show_accepts_pil():
    r = E.run_python('show(Image.new("RGB", (32, 32), (255, 0, 0)))')
    assert r.ok and len(r.images) == 1


def test_show_respects_image_cap():
    r = E.run_python('\n'.join(f'show(np.zeros((30,30,3)))' for _ in range(5)),
                     max_images=2)
    assert r.ok and len(r.images) == 2
    assert 'image limit' in r.stdout


def test_context_preamble_seeds_image(tmp_path):
    from PIL import Image
    p = tmp_path / 'frame.png'
    Image.new('RGB', (1600, 900), (10, 20, 30)).save(p)
    pre = E.build_context_preamble(image_path=str(p),
                                   intrinsics=[[1266.0, 0, 816.0], [0, 1266.0, 491.0], [0, 0, 1]])
    r = E.run_python('print(W, H, int(K[0,0]))', context_preamble=pre)
    assert r.ok and r.stdout.strip() == '1600 900 1266'


def test_as_text_layout():
    r = E.run_python('print("hello"); show(np.zeros((30,30,3)))')
    text = r.as_text()
    assert 'stdout:' in text and 'hello' in text
    assert 'Images:\n<image>' in text
