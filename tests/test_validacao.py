"""
Testes rápidos (malhas grossas, < 1 min no total) que rodam a cada push no GitHub Actions.
Verificam que os solvers convergem e reproduzem as referências dentro de tolerâncias
compatíveis com a malha. As validações completas ficam em cavity.py e pipe.py.
"""
import numpy as np

import ghia
from cavity import CavitySolver
from fvm3d import shift
from pipe import analyze, build


def test_shift():
    a = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(shift(a, 1, 1, -1), [[1, 2, -1], [4, 5, -1]])
    np.testing.assert_array_equal(shift(a, 0, -1, 9), [[9, 9, 9], [0, 1, 2]])


def test_cavidade_2d_re100():
    s = CavitySolver(Re=100, N=32, tol=1e-6)
    assert s.run(max_iter=3000, report=0)
    r = s.compare_ghia()
    assert abs(r['psi_err_pct']) < 2.0                 # 0,25 % em 64x64, 0,05 % em 128x128
    assert r['u_max'] < 0.02 and r['v_max'] < 0.03
    vx = r['vortex']
    assert abs(vx['x'] - ghia.VORTEX[100]['x']) < 0.02 and abs(vx['y'] - ghia.VORTEX[100]['y']) < 0.02


def test_cavidade_2d_conserva_massa():
    s = CavitySolver(Re=400, N=32, tol=1e-6)
    s.run(max_iter=3000, report=0)
    u, v, dx = s.u, s.v, s.dx
    div = (u[1:] - u[:-1]) * dx + (v[:, 1:] - v[:, :-1]) * dx
    assert np.abs(div).max() < 1e-6


def test_tubo_3d_poiseuille():
    s = build(n=9, Re=100, L=12, nz_per_D=6)
    assert s.run(max_iter=1000, report=0)
    res, _ = analyze(s, Re=100, L=12)
    assert abs(res['Q_ratio'] - 1) < 1e-8              # vazão conservada exatamente
    assert abs(res['wc_err_pct']) < 2.5                # -0,8 % já com 11 células no diâmetro
    assert abs(res['fRe_err_pct']) < 4.0
    assert res['prof_rms'] < 0.01
