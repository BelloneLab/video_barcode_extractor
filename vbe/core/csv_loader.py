"""Robust loader for reference traces stored as CSV/TXT/TSV.

Many ephys / DAQ tools emit traces with comment headers, mixed separators,
or metadata preambles. This loader tries several strategies before giving up.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def load_csv_robust(path: str) -> pd.DataFrame:
    """Best-effort CSV loader. Raises the last underlying exception on failure."""
    try:
        lines = Path(path).read_text(encoding='utf-8-sig', errors='replace').splitlines()
        for sep in [',', '\t', ';']:
            for i, line in enumerate(lines[:250]):
                stripped = line.strip()
                if not stripped:
                    continue
                meta_test = stripped.strip('"').lstrip()
                if meta_test.startswith('#'):
                    continue
                parts = [p.strip().strip('"') for p in stripped.split(sep)]
                if len(parts) < 2:
                    continue
                if _looks_numeric(parts[0]):
                    continue
                if not any(any(ch.isalpha() for ch in p) for p in parts):
                    continue
                df = pd.read_csv(path, sep=sep, skiprows=i)
                df = df.loc[:, ~df.columns.astype(str).str.match(r'^Unnamed')]
                if df.shape[1] >= 2 and len(df) > 0:
                    return df
    except Exception:
        pass

    strategies = [
        dict(comment='#', sep=None, engine='python'),
        dict(comment='#', sep=','),
        dict(comment='#', sep=';'),
        dict(comment='#', sep='\t'),
        dict(on_bad_lines='skip', sep=None, engine='python'),
    ]
    last_exc = None
    for kw in strategies:
        try:
            df = pd.read_csv(path, **kw)
            if df.shape[1] >= 1 and len(df) > 0:
                return df
        except Exception as exc:
            last_exc = exc
    raise last_exc if last_exc else RuntimeError(f'Could not parse {path}')
