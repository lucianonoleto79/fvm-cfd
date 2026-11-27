"""
Testes rápidos (malhas grossas, < 1 min no total) que rodam a cada push no GitHub Actions.
Verificam que os solvers convergem e reproduzem as referências dentro de tolerâncias
compatíveis com a malha. As validações completas ficam em cavity.py e pipe.py.
"""
import numpy as np

import ghia
from cavity import CavitySolver
from fvm3d import shift
from monitor import ConvergenceMonitor
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


def test_monitor_curva_e_terminal(tmp_path, capsys):
    png = tmp_path / 'conv.png'
    mon = ConvergenceMonitor(['u', 'v', 'massa'], tol=1e-6, report=50, plot_file=png,
                             refresh=0.0, title='teste')
    s = CavitySolver(Re=100, N=16, tol=1e-6)
    assert s.run(max_iter=2000, monitor=mon)
    out = capsys.readouterr().out
    assert 'max/tol' in out and 'convergiu' in out
    assert f'{s.iter:6d}' in out                       # última iteração impressa
    assert png.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    assert not (tmp_path / 'conv.png.tmp').exists()


def test_tubo_3d_poiseuille():
    s = build(n=9, Re=100, L=12, nz_per_D=6)
    assert s.run(max_iter=1000, report=0)
    res, _ = analyze(s, Re=100, L=12)
    assert abs(res['Q_ratio'] - 1) < 1e-8              # vazão conservada exatamente
    assert abs(res['wc_err_pct']) < 2.5                # -0,8 % já com 11 células no diâmetro
    assert abs(res['fRe_err_pct']) < 4.0
    assert res['prof_rms'] < 0.01


# --- cilindro 2D transiente (cylinder2d.py) ---------------------------------

def test_cilindro_poisson_dct_igual_ao_lu_direto():
    import scipy.sparse.linalg as spla
    from cylinder2d import PoissonDCT
    nx, ny, h = 24, 16, 0.1
    b = np.random.default_rng(0).standard_normal((nx, ny))
    ref = spla.spsolve(PoissonDCT.matrix(nx, ny, h).tocsc(), b.ravel()).reshape(nx, ny)
    np.testing.assert_allclose(PoissonDCT(nx, ny, h).solve(b), ref, rtol=1e-10, atol=1e-12)


def test_cilindro_fracao_solida_area():
    from cylinder2d import solid_fraction
    h = 0.05
    x = np.arange(-1, 1 + h / 2, h)
    X, Y = np.meshgrid(x, x, indexing='ij')
    area = solid_fraction(X, Y, h, 0.5).sum() * h * h
    assert abs(area - np.pi * 0.25) < 2e-3


def test_cilindro_divergencia_nula_e_simetria_re40():
    from cylinder2d import Cylinder2D
    s = Cylinder2D(Re=40, h=0.1, domain=(-4.0, 8.0, -4.0, 4.0), perturbation=0.0)
    r = s.run(tmax=15.0, report=0)
    assert np.abs(s.divergence()).max() < 1e-10       # projeção exata
    assert abs(r['cl_mean']) < 1e-8                   # escoamento simétrico sem perturbação
    assert 1.3 < r['cd_mean'] < 2.5                   # malha grossa e bloqueio de 12,5 %
    assert r['Lw'] > 0.5                              # bolha de recirculação atrás do cilindro
    assert np.isnan(r['St'])                          # permanente: sem frequência de emissão


def test_cilindro_passo_respeita_difusao_ab2():
    from cylinder2d import Cylinder2D
    for Re, h in ((40, 0.04), (100, 0.04), (20, 0.05)):
        s = Cylinder2D(Re=Re, h=h, domain=(-2.0, 4.0, -2.0, 2.0))
        assert s.nu * s.dt / s.h ** 2 <= 1 / 8           # AB2 + difusão explícita 2D


def test_cilindro_strouhal_permanente_e_periodico():
    from cylinder2d import strouhal
    t = np.linspace(0, 100, 5001)
    assert np.isnan(strouhal(t, 1e-3 * np.sin(2 * np.pi * 0.13 * t)))   # ruído de escoamento permanente
    assert abs(strouhal(t, 0.3 * np.sin(2 * np.pi * 0.165 * t)) - 0.165) < 2e-3


def test_cilindro_malha_o_cp_medio_na_parede():
    """Cp médio da malha O referenciado a p_inf: partida a partir do escoamento
    potencial, Cp ~ 1 na estagnação e bem negativo nos flancos."""
    from polar import PolarGrid, PolarNS
    from cylinder import wall_cp_summary
    s = PolarNS(PolarGrid(R0=0.5, Rout=10.0, NT=32), Re=100)
    s.pw_from = 0.0
    s.run(40, monitor=None)
    th, cp = s.wall_cp_mean()
    r = wall_cp_summary(th, cp)
    assert s.pw_n == 40 and th.shape == (32,)
    assert 0.8 < r['Cp_estag'] < 1.3
    assert r['Cp_min'] < -1.0 and 60 < r['ang_Cp_min'] < 120
