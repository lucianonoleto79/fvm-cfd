"""Caso do Lago Paranoá (python/paranoa.py): vazão, conservação e limites do escalar."""
import numpy as np

from monitor import ConvergenceMonitor
from control import run_loop
from paranoa import DIA, VAZOES, LagoGrid, ParanoaFVM


def rodar(h=160.0, dt=3600.0, dias=1.5):
    s = ParanoaFVM(LagoGrid(h), dt=dt)
    s.dias, s.serie = (1,), []
    run_loop(s, int(dias * DIA / dt), ConvergenceMonitor(['a', 'b', 'c'], report=0))
    return s


def test_vazao_divergencia_e_massa():
    s = rodar()
    assert abs(s.vazao_saida() - sum(VAZOES.values())) < 1e-9          # rampa concluída
    assert np.abs(s.divergence()[s.g.wet]).max() * s.dt < 1e-8
    b = s.balanco
    ent = b['entrada_adv'] + b['entrada_dif']
    assert ent > 0
    assert abs(s.massa() + b['saida'] - ent) / ent < 1e-10             # conservação exata


def test_escalar_limitado():
    s = rodar()
    C = s.C[s.g.wet]
    assert C.min() >= -1e-12 and C.max() <= 1 + 1e-12                  # TVD: sem over/undershoot
    assert 1 in s.snaps
