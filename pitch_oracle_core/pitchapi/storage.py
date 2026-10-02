"""Atomic normalized artifact updates with explicit revision keys."""

from pathlib import Path
import pandas as pd

from pitch_oracle_core.pipelines import atomic_output


def read_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    with path.open(encoding="utf-8-sig") as stream:
        header = stream.readline()
    return pd.read_csv(path, sep="\t" if "\t" in header else ",")


def write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    if path.suffix == ".parquet":
        atomic_output(path, lambda temporary: frame.to_parquet(temporary, index=False))
    else:
        atomic_output(path, lambda temporary: frame.to_csv(temporary, sep="\t", index=False))


def append_revisions(frame: pd.DataFrame, path: str | Path, *, keys: list[str]) -> pd.DataFrame:
    existing = read_frame(path)
    if frame.empty:
        return existing
    if set(keys).difference(frame):
        raise ValueError(f"Normalized revisions miss key columns: {set(keys).difference(frame)}")
    incoming = frame.copy()
    if incoming[keys].isna().any().any() or incoming.duplicated(keys).any():
        raise ValueError("Normalized revision keys must be present and unique")
    if not existing.empty:
        # Earlier revisions are immutable: a changed value must have a new observation timestamp.
        overlap = existing.merge(incoming, on=keys, how="inner", suffixes=("_old", "_new"))
        shared = set(existing).intersection(incoming).difference(keys)
        for column in shared:
            old, new = overlap[f"{column}_old"], overlap[f"{column}_new"]
            same = old.eq(new) | (old.isna() & new.isna())
            if not same.all():
                raise ValueError(f"Cannot rewrite normalized revision field {column}")
    result = pd.concat([existing, incoming], ignore_index=True).drop_duplicates(keys, keep="first")
    result = result.sort_values(keys, kind="stable", na_position="last").reset_index(drop=True)
    write_frame(result, path)
    return result
