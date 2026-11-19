import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Tests need a writable temp root.  The default stays inside the checkout, but a
# confined/sandboxed runner (or a read-only checkout) can point OURS_TEST_TMP at a
# writable directory instead of failing on every fixture.
temp = Path(os.environ.get("OURS_TEST_TMP") or (ROOT / "runtime" / "tests"))
temp.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = os.environ["TEMP"] = str(temp)
tempfile.tempdir = str(temp)
os.environ.setdefault("MEMOS_BASE_PATH", str(temp))

# Confined runners (and some Windows endpoint agents) do not grant list access to a
# directory created with mode 0o700, which makes every pytest tmp_path fixture fail
# before the test runs.  When OURS_TEST_TMP is set, create private temp directories
# as 0o777 instead.  Normal (unset) runs keep pytest's default mode.
if os.environ.get("OURS_TEST_TMP"):
    try:  # pragma: no cover - environment shim
        import _pytest.pathlib as _pytest_pathlib

        _original_make_numbered_dir = _pytest_pathlib.make_numbered_dir

        def _make_numbered_dir(root, prefix, mode=0o700):  # noqa: ANN001
            return _original_make_numbered_dir(root, prefix, 0o777)

        _pytest_pathlib.make_numbered_dir = _make_numbered_dir

        _real_mkdir = os.mkdir

        def _mkdir(path, mode=0o777, *args, **kwargs):  # noqa: ANN001
            return _real_mkdir(path, 0o777 if int(mode) == 0o700 else mode, *args, **kwargs)

        os.mkdir = _mkdir
    except Exception:  # noqa: BLE001
        pass

