"""Cached Streamlit facade over the format-aware artifact repository."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

from pitch_oracle_core.artifacts.repository import ArtifactRepository as BaseRepository
from pitch_oracle_core.artifacts.repository import read_tabular_frame


@st.cache_data(show_spinner=False, max_entries=128, ttl=600)
def _read_frame(path: str, modified_ns: int, columns=None, filters=None, delimiter=None) -> pd.DataFrame:
    del modified_ns
    return read_tabular_frame(path, columns=columns, filters=filters, delimiter=delimiter)


@st.cache_data(show_spinner=False)
def _read_json(path: str, modified_ns: int) -> dict:
    del modified_ns
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON artifact must be an object")
    return value


@st.cache_data(show_spinner=False)
def _read_arrays(path: str, modified_ns: int) -> dict[str, np.ndarray]:
    del modified_ns
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class ArtifactRepository(BaseRepository):
    def frame(self, name: str, *, columns=None, filters=None) -> pd.DataFrame:
        path = self.path(name)
        return _read_frame(str(path), path.stat().st_mtime_ns, columns, filters, self.descriptors[name].get("delimiter"))

    def json(self, name: str) -> dict:
        path = self.path(name)
        return _read_json(str(path), path.stat().st_mtime_ns)

    def arrays(self, name: str) -> dict[str, np.ndarray]:
        path = self.path(name)
        return _read_arrays(str(path), path.stat().st_mtime_ns)
