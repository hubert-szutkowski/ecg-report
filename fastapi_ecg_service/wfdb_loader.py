"""Wczytywanie sygnałów EKG z PhysioNet (wfdb) lub bezpośrednich URL do .dat/.hea."""

import tempfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import wfdb


def load_from_physionet(record_name: str, pn_dir: str, channel: int = 0) -> np.ndarray:
    """
    Pobiera rekord bezpośrednio z bazy PhysioNet.
    Przykład: record_name='100', pn_dir='mitdb' -> rekord mitdb/100
    """
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    return record.p_signal[:, channel].astype(np.float32)


def load_from_url(hea_url: str, dat_url: str, channel: int = 0) -> np.ndarray:
    """
    Pobiera pliki .hea i .dat spod dowolnego URL, wczytuje lokalnie przez wfdb.
    Oba pliki muszą mieć tę samą nazwę bazową (np. record.hea, record.dat).
    """
    base_name = Path(urlparse(hea_url).path).stem
    if Path(urlparse(dat_url).path).stem != base_name:
        raise ValueError("Nazwy plików .hea i .dat muszą się zgadzać (ten sam rekord).")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for url, suffix in [(hea_url, ".hea"), (dat_url, ".dat")]:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            (tmp_path / f"{base_name}{suffix}").write_bytes(resp.content)

        record = wfdb.rdrecord(str(tmp_path / base_name))
        return record.p_signal[:, channel].astype(np.float32)