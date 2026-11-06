"""Testes do solver cartesiano com fronteira imersa (domínio retangular)."""
import numpy as np

from cartesian import Annulus, CartesianGrid, CartesianIBNS, Cylinder
from control import RunControl, load_checkpoint
from cylinder import pulse
from monitor import ConvergenceMonitor


def quiet():
    return ConvergenceMonitor(['Cd', 'Cl'], report=0, logy=False)


def couette_ib(n):
    R0, R1 = 0.5, 1.0
    f = np.linspace(-1.2, 1.2, n + 1)
    g = CartesianGrid(f, f)
    s = CartesianIBNS(g, Annulus(R0, R1, 1.0), Re=10, dt=0.01, outer='closed', box=None)
    s.run(800, monitor=quiet())
    A = R0 / (R0 ** 2 - R1 ** 2)
    ex = lambda X, Y: A * np.hypot(X, Y) - A * R1 ** 2 / np.hypot(X, Y)
    UX, UY = np.meshgrid(g.xf, g.yc, indexing='ij')
    VX, VY = np.meshgrid(g.xc, g.yf, indexing='ij')
    fu, fv = s.body.phi(UX, UY) > 0, s.body.phi(VX, VY) > 0
    eu = (s.u + ex(UX, UY) * UY / np.hypot(UX, UY))[fu]
    ev = (s.v - ex(VX, VY) * VX / np.hypot(VX, VY))[fv]
    return np.sqrt(np.mean(np.r_[eu, ev] ** 2)), s


def test_fronteira_imersa_taylor_couette_converge():
    e1, _ = couette_ib(40)
    e2, s = couette_ib(80)
    assert e2 < 4e-3 and e1 / e2 > 2.5                 # ordem >= ~1,3 (medido: 3,5)
    XC, YC = np.meshgrid(s.g.xc, s.g.yc, indexing='ij')
    assert np.abs(s.divergence()[s.body.phi(XC, YC) > 0]).max() < 1e-12


def test_malha_mesma_perto_do_corpo():
    a = CartesianGrid.for_cylinder(10, 20, 10)
    b = CartesianGrid.for_cylinder(40, 80, 40)
    na, nb = (np.abs(a.xf) <= 9.5), (np.abs(b.xf) <= 9.5)
    np.testing.assert_allclose(a.xf[na], b.xf[nb])     # domínio maior só acrescenta células externas
    np.testing.assert_allclose(a.yf[np.abs(a.yf) <= 9], b.yf[np.abs(b.yf) <= 9])
    assert np.isclose(a.dx.min(), 0.025)


def test_retangular_retomada_exata(tmp_path):
    def new():
        return CartesianIBNS(CartesianGrid.for_cylinder(5, 8, 4, h=0.1, core=1.5), Cylinder(0.5, pulse), Re=100)
    ref = new(); ref.run(40, monitor=quiet())
    a = new()
    a.run(20, monitor=quiet(), control=RunControl(tmp_path / 'ck.npz', 0, None, keyboard=False, verbose=False))
    b = load_checkpoint(new(), tmp_path / 'ck.npz')
    b.run(40, monitor=quiet())
    np.testing.assert_allclose(b.u, ref.u, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.array(b.history), np.array(ref.history), rtol=1e-9, atol=1e-11)


def test_retangular_simetrico_sem_perturbacao():
    s = CartesianIBNS(CartesianGrid.for_cylinder(5, 8, 4, h=0.1, core=1.5), Cylinder(0.5), Re=100)
    s.run(60, monitor=quiet())
    cd, cl = s.history[-1][1:]
    assert abs(cl) < 1e-8 and 0.8 < cd < 3.0
