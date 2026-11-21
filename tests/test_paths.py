"""Tests for the default locations derived from the ``MEMORILLA_*`` environment variables."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from memorilla import paths


def load_paths() -> ModuleType:
    """Evaluate a fresh copy of ``memorilla.paths`` under the current environment.

    The copy is not registered in ``sys.modules``, so the imported module and its users are left untouched.

    Returns:
        The freshly evaluated module.
    """
    module = importlib.util.module_from_spec(paths.__spec__)
    paths.__spec__.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative", [True, False], ids=["relative-home", "absolute-home"])
def test_default_dirs_under_home(relative: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default data and results directories sit directly under ``MEMORILLA_HOME``, relative or absolute."""
    home = Path("work") if relative else tmp_path / "work"
    # An empty working directory, so no relative path refers to an existing directory.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORILLA_HOME", str(home))
    monkeypatch.delenv("MEMORILLA_DATA_DIR", raising=False)
    module = load_paths()
    defaults = {"data": module.data_root("memorilla/Memorilla-Data"), "results": module.resolve(module.RESULTS_DIR)}
    assert defaults == {"data": home / "data", "results": home / "results"}
