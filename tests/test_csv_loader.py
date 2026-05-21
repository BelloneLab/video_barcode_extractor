"""Unit tests for the robust CSV/TXT loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vbe.core.csv_loader import load_csv_robust


def _write(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content, encoding='utf-8')
    return str(p)


def test_loads_comma_csv(tmp_path):
    path = _write(tmp_path, 'a.csv',
                  'time,signal\n0.0,1\n0.1,2\n0.2,3\n')
    df = load_csv_robust(path)
    assert list(df.columns) == ['time', 'signal']
    assert len(df) == 3


def test_loads_semicolon_csv(tmp_path):
    path = _write(tmp_path, 'b.csv',
                  'time;signal;ttl\n0.0;1;0\n0.1;2;1\n0.2;3;0\n')
    df = load_csv_robust(path)
    assert list(df.columns) == ['time', 'signal', 'ttl']
    assert len(df) == 3


def test_loads_csv_with_comment_preamble(tmp_path):
    content = (
        '# recorded with rig X\n'
        '# date: 2026-05-01\n'
        'time,value\n'
        '0.000,3.21\n'
        '0.001,3.22\n'
    )
    path = _write(tmp_path, 'c.csv', content)
    df = load_csv_robust(path)
    assert list(df.columns) == ['time', 'value']
    assert len(df) == 2


def test_loads_tab_csv(tmp_path):
    path = _write(tmp_path, 'd.tsv',
                  'time\tsignal\n0.0\t1\n0.1\t2\n')
    df = load_csv_robust(path)
    assert list(df.columns) == ['time', 'signal']


def test_raises_on_completely_empty_file(tmp_path):
    path = _write(tmp_path, 'empty.csv', '')
    with pytest.raises(Exception):
        load_csv_robust(path)
