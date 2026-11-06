"""Testes do solver transiente em malha polar (cilindro)."""
import numpy as np

from control import RunControl, load_checkpoint
from cylinder import pulse, stats
from monitor import ConvergenceMonitor
from polar import PolarGrid, PolarNS


def quiet():
    return ConvergenceMonitor(['Cd', 'Cl'], report=0, logy=False)


def taylor_couette(NT):
    R0, R1 = 0.5, 1.0
    g = PolarGrid(R0=R0, Rout=R1, NT=NT, aspect=1.0)
    s = PolarNS(g, Re=10, dt=0.02, outer=('wall', 0.0), wall_speed=lambda t: 1.0)
    s.run(600, monitor=quiet())
    A = R0 / (R0 ** 2 - R1 ** 2)
    exact = A * g.rc - A * R1 ** 2 / g.rc
    return np.abs(s.ut - exact[:, None]).max(), s


def test_taylor_couette_segunda_ordem():
    e1, _ = taylor_couette(32)
    e2, s = taylor_couette(64)
    assert e2 < 3e-3 and e1 / e2 > 3.0                 # ~2ª ordem (9,0e-3 -> 2,4e-3)
    assert np.abs(s.ur).max() < 1e-12                  # sem escoamento radial espúrio
    assert np.abs(s.divergence()).max() < 1e-12


def test_malha_o():
    g = PolarGrid(R0=0.5, Rout=40, NT=128, aspect=0.7)
    assert np.isclose(g.rf[0], 0.5) and np.isclose(g.rf[-1], 40)
    ratio = g.drp / (g.rc * g.dth)                    # células quase quadradas em toda a malha
    assert np.allclose(ratio, ratio[0], rtol=1e-2) and 0.6 < ratio[0] < 0.8


def test_cilindro_simetrico_sem_perturbacao():
    g = PolarGrid(R0=0.5, Rout=10, NT=64)
    s = PolarNS(g, Re=100)
    s.run(150, monitor=quiet())
    cd, cl = s.history[-1][1:]
    assert abs(cl) < 1e-8 and 1.0 < cd < 3.0           # sem perturbação: escoamento simétrico
    assert np.abs(s.divergence()).max() < 1e-11


def test_cilindro_retomada_exata(tmp_path):
    def new():
        return PolarNS(PolarGrid(R0=0.5, Rout=10, NT=48), Re=100, wall_speed=pulse)
    ref = new()
    ref.run(80, monitor=quiet())
    a = new()
    a.run(40, monitor=quiet(), control=RunControl(tmp_path / 'ck.npz', 0, None, keyboard=False,
                                                    verbose=False))
    b = load_checkpoint(new(), tmp_path / 'ck.npz')
    b.run(80, monitor=quiet())
    np.testing.assert_allclose(b.ut, ref.ut, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.array(b.history), np.array(ref.history), rtol=1e-10, atol=1e-12)


def test_strouhal_de_sinal_sintetico():
    class S:
        dt = 0.01
    s = S()
    t = np.arange(1, 20001) * s.dt
    s.history = [(k, 1.33 + 0.01 * np.sin(4 * np.pi * 0.165 * tt), 0.33 * np.sin(2 * np.pi * 0.165 * tt))
                 for k, tt in zip(range(1, 20001), t)]
    r = stats(s, 50)
    assert abs(r['St'] - 0.165) < 1e-4 and abs(r['Cl_amp'] - 0.33) < 1e-3
    assert abs(r['Cd_mean'] - 1.33) < 1e-3 and abs(r['Cd_amp'] - 0.01) < 1e-3
