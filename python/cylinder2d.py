"""
cylinder2d.py — Escoamento 2D TRANSIENTE em torno de um cilindro circular (FVM)
-------------------------------------------------------------------------------
Equações (adimensionais, rho = 1, U = D = 1, nu = 1/Re):
    div(u) = 0
    du/dt + div(u u) = -grad(p) + (1/Re) lap(u) + f        (f: fronteira imersa)

Método:
  - Volumes finitos em malha cartesiana uniforme DESENCONTRADA (como cavity.py):
    p no centro das células, u nas faces verticais, v nas horizontais.
  - Passo fracionado (projeção incremental, Kim & Moin 1985 / Van Kan 1986):
      1. u* = u^n + dt [ AB2(-conv + visc) - grad p^n ]      (explícito)
      2. fronteira imersa: u** = (1 - alpha) u*,  f = (u** - u*)/dt
      3. lap(phi) = div(u**)/dt
      4. u^{n+1} = u** - dt grad(phi),   p^{n+1} = p^n + phi
  - Convecção: forma conservativa com QUICK (Leonard 1979, padrão) ou
    diferenças centrais; difusão: diferenças centrais. Adams-Bashforth 2.
  - Cilindro: fronteira imersa por fração de volume (Kajishima et al. 2001):
    alpha = fração da célula de cada componente da velocidade ocupada pelo
    sólido. A força do fluido sobre o corpo é F = -sum(f) dV.
  - Poisson: transformada de cosseno (DCT-II) em y (paredes de Neumann) e,
    para cada modo, sistema tridiagonal em x fatorado uma única vez (LU
    esparso). É um solver direto: o mesmo resultado do LU completo, bem mais
    barato.

Contornos (domínio [x0,x1] x [y0,y1], cilindro de diâmetro D na origem):
    entrada x0: u = 1, v = 0          topo/fundo: deslizamento (v = 0, du/dy = 0)
    saída  x1: convectiva  du/dt + U du/dx = 0,  p = 0 (phi = 0 na face)
Perturbação inicial em v (a mesma do Cordel, casos/cylinder.py) para disparar
a emissão de vórtices.

Arranjo dos arrays (índice [i, j], i em x, j em y), h = espaçamento:
    p : (nx,   ny)    p[i,j] em (x0+(i+.5)h, y0+(j+.5)h)
    u : (nx+1, ny)    u[i,j] em (x0+i h,     y0+(j+.5)h)
    v : (nx,   ny+1)  v[i,j] em (x0+(i+.5)h, y0+j h)

Uso:
    python cylinder2d.py                                # Re = 100, h = 0,04, t = 200
    python cylinder2d.py --Re 40 --tmax 100
    python cylinder2d.py --Re 100 --h 0.08 0.04 0.03333 # estudo de malha
    python cylinder2d.py --Re 100 --h 0.04 --replot     # refaz figura e json do campos.npz

Saídas (convenção de casefiles.py): results/cilindro_mac_Re<Re>_h<h>/ com
04_resultados.png, resultados.json, cp_parede.csv e campos.npz (não versionado);
o resumo dos casos de um Re, em results/estudo_cilindro_mac_Re<Re>/resultados.json.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from casefiles import CaseDir, study_dir
import scipy.sparse.linalg as spla
from scipy.fft import dct, idct
from scipy.ndimage import map_coordinates

DEFAULT_OUT = Path(__file__).resolve().parent.parent / 'results'


def solid_fraction(X, Y, h, R, nsub=10):
    """Fração de área de cada célula h x h (centrada em X, Y) dentro do círculo
    de raio R na origem; subamostragem nsub x nsub só na faixa da interface."""
    r = np.hypot(X, Y)
    alpha = (r < R).astype(float)
    band = np.abs(r - R) < h
    s = ((np.arange(nsub) + 0.5) / nsub - 0.5) * h
    ox, oy = np.meshgrid(s, s, indexing='ij')
    xb, yb = X[band][:, None], Y[band][:, None]
    alpha[band] = (np.hypot(xb + ox.ravel(), yb + oy.ravel()) < R).mean(1)
    return alpha


def face_values(q, a, axis, scheme):
    """Valores nas faces entre q[k] e q[k+1] (k = 1..M-3) de um array com uma
    célula fantasma de cada lado em 'axis' (M pontos): QUICK a montante do
    sentido de a, ou média (centrada)."""
    M = q.shape[axis]
    L2, L, R, R2 = (np.take(q, np.arange(s, M - 3 + s), axis=axis) for s in range(4))
    if scheme == 'central':
        return 0.5 * (L + R)
    return np.where(a >= 0, 0.75 * L + 0.375 * R - 0.125 * L2, 0.75 * R + 0.375 * L - 0.125 * R2)


CL_AMP_PERMANENTE = 0.02   # abaixo disso Cl não oscila: escoamento permanente (= CL_MIN do Cordel)


def strouhal(t, cl):
    """Frequência dominante de Cl (FFT com zero-padding) -> St = f D / U
    (mesma rotina de casos/cylinder.py do Cordel). Devolve nan se o escoamento
    for permanente (amplitude de Cl < CL_AMP_PERMANENTE): o pico da FFT seria
    só ruído numérico, sem significado físico."""
    if len(t) < 8 or np.ptp(cl) / 2 < CL_AMP_PERMANENTE:
        return float('nan')
    ti = np.linspace(t[0], t[-1], 4096)
    c = np.interp(ti, t, cl)
    c = (c - c.mean()) * np.hanning(len(c))
    F = np.abs(np.fft.rfft(c, 8 * len(c)))
    f = np.fft.rfftfreq(8 * len(c), ti[1] - ti[0])
    return float(f[1:][np.argmax(F[1:])])


def fmt_st(st, nd=4):
    """St formatado; '—' (com largura fixa) quando o escoamento é permanente."""
    return f'{st:.{nd}f}' if np.isfinite(st) else '—'.center(nd + 2)


class PoissonDCT:
    """lap(phi) = b nas células: Neumann em x0, y0, y1; Dirichlet (phi = 0 na
    face) em x1. DCT-II ortonormal em y diagonaliza o operador de Neumann
    (autovalores -4 sen^2(pi k / 2 ny) / h^2); sobra um tridiagonal em x por modo."""

    def __init__(self, nx, ny, h):
        self.nx, self.ny = nx, ny
        d = -2.0 * np.ones(nx)
        d[0], d[-1] = -1.0, -3.0
        Lx = sp.diags([d, np.ones(nx - 1), np.ones(nx - 1)], [0, 1, -1]) / h ** 2
        mu = -4.0 * np.sin(np.pi * np.arange(ny) / (2 * ny)) ** 2 / h ** 2
        A = sp.kron(sp.eye(ny), Lx) + sp.diags(np.repeat(mu, nx))
        self.lu = spla.splu(A.tocsc())

    def solve(self, b):
        bh = dct(b, type=2, axis=1, norm='ortho')
        ph = self.lu.solve(bh.T.ravel()).reshape(self.ny, self.nx).T
        return idct(ph, type=2, axis=1, norm='ortho')

    @staticmethod
    def matrix(nx, ny, h):
        """Matriz 5 pontos do mesmo operador (para o teste de verificação)."""
        def lap1(n, dirichlet_right):
            d = -2.0 * np.ones(n)
            d[0], d[-1] = -1.0, (-3.0 if dirichlet_right else -1.0)
            return sp.diags([d, np.ones(n - 1), np.ones(n - 1)], [0, 1, -1])
        return (sp.kron(lap1(nx, True), sp.eye(ny)) + sp.kron(sp.eye(nx), lap1(ny, False))) / h ** 2


class Cylinder2D:
    def __init__(self, Re=100.0, h=0.04, domain=(-6.0, 16.0, -7.0, 7.0), cfl=0.4, dt=None,
                 scheme='quick', perturbation=0.05, t_mean=None):
        x0, x1, y0, y1 = domain
        self.nx, self.ny = int(round((x1 - x0) / h)), int(round((y1 - y0) / h))
        self.h = h = (x1 - x0) / self.nx
        y1 = y0 + self.ny * h
        self.domain = (x0, x1, y0, y1)
        self.Re, self.nu, self.U, self.R = Re, 1.0 / Re, 1.0, 0.5
        self.scheme = scheme
        # passo de tempo: CFL convectivo (u_max ~ 1,6 U) e limite da difusão
        # explícita com AB2 em 2D: autovalor -8 nu/h^2, AB2 estável no eixo real
        # até lambda dt = -1  ->  nu dt / h^2 <= 1/8 (usa 0,1 com margem)
        self.dt = dt or min(cfl * h / 1.6, 0.1 * h * h / self.nu)
        self.t_mean = t_mean
        nx, ny = self.nx, self.ny
        self.xc = x0 + (np.arange(nx) + 0.5) * h
        self.yc = y0 + (np.arange(ny) + 0.5) * h
        self.xf = x0 + np.arange(nx + 1) * h
        self.yf = y0 + np.arange(ny + 1) * h
        Xu, Yu = np.meshgrid(self.xf, self.yc, indexing='ij')
        Xv, Yv = np.meshgrid(self.xc, self.yf, indexing='ij')
        self.au = solid_fraction(Xu, Yu, h, self.R)
        self.av = solid_fraction(Xv, Yv, h, self.R)
        # condição inicial: partida impulsiva + perturbação em v (como no Cordel)
        self.u = np.full((nx + 1, ny), self.U)
        self.v = perturbation * np.exp(-((Xv - 2.0) ** 2 + Yv ** 2))
        self.v[:, 0] = self.v[:, -1] = 0.0
        self.p = np.zeros((nx, ny))
        self.Hu_old = self.Hv_old = None
        self.poisson = PoissonDCT(nx, ny, h)
        self.t, self.it = 0.0, 0
        self.hist = []                       # (t, Cd, Cl)
        # pontos de amostragem da pressão na parede (r = R), theta a partir de +x
        self.theta = np.linspace(0, 2 * np.pi, 181)[:-1]
        self.cp_sum, self.cp_n = np.zeros_like(self.theta), 0

    # --- termos explícitos -------------------------------------------------
    def rhs(self):
        """H = -conv + nu lap, nos nós internos de u (i = 1..nx-1) e v (j = 1..ny-1)."""
        u, v, h, U, sc = self.u, self.v, self.h, self.U, self.scheme
        # u: fluxos em x nos centros das células
        a = 0.5 * (u[:-1] + u[1:])
        up = np.concatenate([np.full((1, self.ny), U), u, u[-1:]], 0)
        Fx = a * face_values(up, a, 0, sc)
        # u: fluxos em y nos cantos (paredes de deslizamento: fluxo nulo)
        av = 0.5 * (v[:-1, :] + v[1:, :])
        ui = u[1:-1]
        uyp = np.concatenate([ui[:, :1], ui, ui[:, -1:]], 1)
        Fy = np.zeros_like(av)
        Fy[:, 1:-1] = av[:, 1:-1] * face_values(uyp, av[:, 1:-1], 1, sc)
        conv_u = (Fx[1:] - Fx[:-1]) / h + (Fy[:, 1:] - Fy[:, :-1]) / h
        lap_u = (u[2:] - 2 * ui + u[:-2]) / h ** 2 + (uyp[:, 2:] - 2 * ui + uyp[:, :-2]) / h ** 2
        Hu = -conv_u + self.nu * lap_u
        # v: fluxos em y nos centros das células
        a = 0.5 * (v[:, :-1] + v[:, 1:])
        vp = np.concatenate([-v[:, 1:2], v, -v[:, -2:-1]], 1)
        Gy = a * face_values(vp, a, 1, sc)
        # v: fluxos em x nos cantos (entrada: v = 0; saída: gradiente nulo)
        au = 0.5 * (u[:, :-1] + u[:, 1:])
        vi = v[:, 1:-1]
        vxp = np.concatenate([-vi[:1], vi, vi[-1:]], 0)
        Gx = np.zeros_like(au)
        Gx[1:-1] = au[1:-1] * face_values(vxp, au[1:-1], 0, sc)
        Gx[-1] = au[-1] * vi[-1]
        conv_v = (Gx[1:] - Gx[:-1]) / h + (Gy[:, 1:] - Gy[:, :-1]) / h
        lap_v = (vxp[2:] - 2 * vi + vxp[:-2]) / h ** 2 + (v[:, 2:] - 2 * vi + v[:, :-2]) / h ** 2
        Hv = -conv_v + self.nu * lap_v
        return Hu, Hv

    def step(self):
        h, dt = self.h, self.dt
        u, v, p = self.u, self.v, self.p
        Hu, Hv = self.rhs()
        if self.Hu_old is None:
            Au, Av = Hu, Hv                                   # Euler no 1º passo
        else:
            Au, Av = 1.5 * Hu - 0.5 * self.Hu_old, 1.5 * Hv - 0.5 * self.Hv_old
        self.Hu_old, self.Hv_old = Hu, Hv
        us, vs = u.copy(), v.copy()
        us[1:-1] += dt * (Au - (p[1:] - p[:-1]) / h)
        vs[:, 1:-1] += dt * (Av - (p[:, 1:] - p[:, :-1]) / h)
        us[-1] = u[-1] - dt * self.U * (u[-1] - u[-2]) / h      # saída convectiva
        # fronteira imersa (fração de volume): u** = (1 - alpha) u*
        fu, fv = -self.au * us / dt, -self.av * vs / dt
        us += dt * fu
        vs += dt * fv
        Fx, Fy = -fu.sum() * h * h, -fv.sum() * h * h       # força do fluido sobre o corpo
        # projeção
        div = (us[1:] - us[:-1] + vs[:, 1:] - vs[:, :-1]) / h
        phi = self.poisson.solve(div / dt)
        us[1:-1] -= dt * (phi[1:] - phi[:-1]) / h
        us[-1] -= dt * (-2.0 * phi[-1]) / h
        vs[:, 1:-1] -= dt * (phi[:, 1:] - phi[:, :-1]) / h
        self.u, self.v, self.p = us, vs, p + phi
        self.t += dt
        self.it += 1
        coef = 2.0 / (self.U ** 2 * 2 * self.R)
        self.hist.append((self.t, coef * Fx, coef * Fy))
        if self.t_mean is not None and self.t >= self.t_mean:
            self.cp_sum += self.wall_pressure()
            self.cp_n += 1

    # --- pós-processamento -------------------------------------------------
    def interp_cells(self, f, x, y):
        """Interpolação bilinear de um campo nos centros das células."""
        x0, _, y0, _ = self.domain
        ix = (np.asarray(x) - x0) / self.h - 0.5
        iy = (np.asarray(y) - y0) / self.h - 0.5
        return map_coordinates(f, [ix, iy], order=1, mode='nearest')

    def wall_pressure(self):
        return self.interp_cells(self.p, self.R * np.cos(self.theta), self.R * np.sin(self.theta))

    def divergence(self):
        return (self.u[1:] - self.u[:-1] + self.v[:, 1:] - self.v[:, :-1]) / self.h

    def vorticity(self):
        """omega = dv/dx - du/dy nos cantos internos (x0 + i h, y0 + j h), i, j >= 1."""
        h = self.h
        return (self.v[1:, 1:-1] - self.v[:-1, 1:-1]) / h - (self.u[1:-1, 1:] - self.u[1:-1, :-1]) / h

    def cell_velocity(self):
        return 0.5 * (self.u[:-1] + self.u[1:]), 0.5 * (self.v[:, :-1] + self.v[:, 1:])

    def recirculation_length(self):
        """L_w/D: do ponto de trás do cilindro (x = R) até onde u volta a ser
        positivo na linha de centro (y = 0); 0 se não há recirculação."""
        uc, _ = self.cell_velocity()
        x = np.linspace(self.R, self.domain[1] - 1, 4000)
        ul = self.interp_cells(uc, x, np.zeros_like(x))
        neg = ul < 0
        if not neg[:20].any():
            return 0.0
        k = np.argmax(~neg & (np.arange(len(x)) > np.argmax(neg)))
        return float(x[k] - self.R) / (2 * self.R)

    def run(self, tmax, report=500, wall_limit=None):
        t0 = time.time()
        if self.t_mean is None:
            self.t_mean = tmax / 2
        while self.t < tmax - 1e-12:
            self.step()
            if not np.isfinite(self.hist[-1][1]):
                raise FloatingPointError(f'divergiu em t = {self.t:.3f}')
            if report and self.it % report == 0:
                _, cd, cl = self.hist[-1]
                print(f'  it {self.it:7d}  t = {self.t:8.3f}  Cd = {cd:7.4f}  Cl = {cl:+7.4f}  '
                      f'max|div| = {np.abs(self.divergence()).max():.1e}  {time.time() - t0:7.0f} s',
                      flush=True)
        self.seconds = time.time() - t0
        return self.summary()

    def summary(self):
        H = np.array(self.hist)
        t, cd, cl = H.T
        sel = t >= self.t_mean
        cdm, clm = cd[sel].mean(), cl[sel].mean()
        r = dict(Re=self.Re, h=self.h, D_over_h=2 * self.R / self.h, nx=self.nx, ny=self.ny,
                 cells=self.nx * self.ny, dt=self.dt, tmax=float(t[-1]), t_mean=self.t_mean,
                 steps=self.it, scheme=self.scheme, domain=self.domain,
                 cd_mean=float(cdm), cl_mean=float(clm),
                 cl_rms=float(np.sqrt(np.mean((cl[sel] - clm) ** 2))),
                 cl_amp=float(np.ptp(cl[sel]) / 2),
                 cd_rms=float(np.sqrt(np.mean((cd[sel] - cdm) ** 2))),
                 St=strouhal(t[sel], cl[sel]),
                 Lw=self.recirculation_length(),
                 seconds=float(getattr(self, 'seconds', 0.0)))
        if self.cp_n:
            cp = 2 * self.cp_sum / self.cp_n / self.U ** 2
            ang = np.degrees(np.abs(np.pi - self.theta))     # 0 = estagnação
            r.update(cp_stag=float(cp[np.argmin(ang)]), cp_min=float(cp.min()),
                     theta_cp_min=float(ang[np.argmin(cp)]), cp_base=float(cp[np.argmax(ang)]))
        return r

    def save(self, fname):
        H = np.array(self.hist)
        uc, vc = self.cell_velocity()
        cp = 2 * self.cp_sum / max(self.cp_n, 1) / self.U ** 2
        np.savez(fname, t=H[:, 0], cd=H[:, 1], cl=H[:, 2], theta=self.theta, cp_medio=cp,
                 xc=self.xc, yc=self.yc, xf=self.xf, yf=self.yf, u=uc, v=vc, p=self.p,
                 w=self.vorticity(), resumo=json.dumps(self.summary()))


def plot_case(npz, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    d = np.load(npz)
    r = json.loads(str(d['resumo']))
    t, cd, cl = d['t'], d['cd'], d['cl']
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    ax = fig.add_subplot(gs[0, :])
    lim = 3.0
    cs = ax.contourf(d['xf'][1:-1], d['yf'][1:-1], np.clip(d['w'], -lim, lim).T,
                     levels=np.linspace(-lim, lim, 41), cmap='RdBu_r', extend='both')
    ax.add_patch(Circle((0, 0), 0.5, fc='white', ec='#0b0b0b'))
    ax.set(xlim=(-3, min(15, d['xf'][-1])), ylim=(-4, 4), aspect='equal',
           title=f"Vorticidade em t = {t[-1]:.0f}, Re = {r['Re']:g} "
                 f"(h = D/{r['D_over_h']:.0f}, {r['nx']}x{r['ny']} células)")
    fig.colorbar(cs, ax=ax, shrink=0.8, label='vorticidade ω D/U')
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, cd, color='#2a78d6', lw=1.2, label='Cd')
    ax.plot(t, cl, color='#eb6834', lw=1.2, label='Cl')
    ax.set(xlabel='t U / D', ylabel='coeficiente',
           title=f"Cd = {r['cd_mean']:.3f}, Cl' RMS = {r['cl_rms']:.3f}, St = {fmt_st(r['St'], 3).strip()}")
    sel = t > min(5.0, t[-1] / 2)
    ax.set_ylim(np.percentile(np.r_[cd[sel], cl[sel]], [0.5, 99.5]) + np.array([-0.2, 0.2]))
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    ax = fig.add_subplot(gs[1, 1])
    ang = np.degrees(np.abs(np.pi - d['theta']))
    o = np.argsort(ang)
    a = np.linspace(0, 180, 181)
    ax.plot(a, 1 - 4 * np.sin(np.radians(a)) ** 2, '--', color='#6b6b6b', lw=1.2, label='potencial')
    ax.plot(ang[o], d['cp_medio'][o], '.', color='#2a78d6', ms=4, label='FVM (média temporal)')
    ax.set(xlabel='θ a partir do ponto de estagnação (graus)', ylabel='Cp', xlim=(0, 180),
           title='Cp médio na parede')
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def _sem_nan(r):
    """JSON padrão: nan -> None (a tabela do README mostra '—')."""
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}


def finish_case(case):
    """Figura, Cp na parede (CSV) e resultados.json de um caso a partir do campos.npz."""
    d = np.load(case.campos)
    r = json.loads(str(d['resumo']))
    plot_case(case.campos, case.resultado())
    ang = np.degrees(np.abs(np.pi - d['theta']))
    np.savetxt(case.root / 'cp_parede.csv', np.c_[d['theta'], ang, d['cp_medio']], delimiter=',',
               fmt='%.6f', header='theta_rad_de_+x,angulo_da_estagnacao_graus,cp_medio', comments='')
    out = _sem_nan(dict(r, solver='cylinder2d', Cd_mean=r['cd_mean'], Cl_amp=r['cl_amp'],
                        Cl_rms=r['cl_rms']))
    case.save_json(out)
    return r


def main():
    ap = argparse.ArgumentParser(description='Cilindro 2D transiente — FVM, passo fracionado, fronteira imersa')
    ap.add_argument('--Re', type=float, nargs='+', default=[100.0])
    ap.add_argument('--h', type=float, nargs='+', default=[0.04], help='espaçamento (D = 1)')
    ap.add_argument('--domain', type=float, nargs=4, default=[-6.0, 16.0, -7.0, 7.0],
                    metavar=('X0', 'X1', 'Y0', 'Y1'))
    ap.add_argument('--tmax', type=float, default=200.0)
    ap.add_argument('--t-mean', type=float, default=None, help='início das médias (padrão: tmax/2)')
    ap.add_argument('--cfl', type=float, default=0.4)
    ap.add_argument('--dt', type=float, default=None)
    ap.add_argument('--scheme', choices=['quick', 'central'], default='quick')
    ap.add_argument('--report', type=int, default=1000)
    ap.add_argument('--replot', action='store_true', help='só refaz figura e json a partir do campos.npz')
    ap.add_argument('--out', default=str(DEFAULT_OUT), help='pasta de saída (padrão: results/)')
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for Re in args.Re:
        for h in args.h:
            case = CaseDir(OUT, f'cilindro_mac_Re{Re:g}_h{h:g}')
            if args.replot:
                finish_case(case)
                continue
            s = Cylinder2D(Re=Re, h=h, domain=args.domain, cfl=args.cfl, dt=args.dt,
                           scheme=args.scheme, t_mean=args.t_mean)
            print(f'\n=== Re = {Re:g}, h = {s.h:.4f} (D/h = {1 / s.h:.1f}), {s.nx}x{s.ny} células, '
                  f'dt = {s.dt:.4g}, {int(np.ceil(args.tmax / s.dt))} passos ===', flush=True)
            r = s.run(args.tmax, report=args.report)
            s.save(case.campos)
            finish_case(case)
            print(json.dumps(r, indent=1, ensure_ascii=False))
            results.append(r)
    if results:
        print('\n  Re     D/h   células   Cd_med   Cl_RMS   St      Lw/D   Cp_estag  Cp_min (ang)  tempo')
        for r in results:
            print(f"{r['Re']:5g} {r['D_over_h']:6.1f} {r['cells']:8d}  {r['cd_mean']:.4f}  {r['cl_rms']:.4f}  "
                  f"{fmt_st(r['St'])}  {r['Lw']:.3f}  {r.get('cp_stag', np.nan):.3f}  "
                  f"{r.get('cp_min', np.nan):.3f} ({r.get('theta_cp_min', np.nan):.0f}°)  {r['seconds']:.0f} s")
        for Re in sorted({r['Re'] for r in results}):       # acumula com execuções anteriores
            est = study_dir(OUT, f'cilindro_mac_Re{Re:g}') / 'resultados.json'
            old = json.load(open(est, encoding='utf-8')) if est.exists() else []
            keep = {r['h']: r for r in old + [_sem_nan(r) for r in results if r['Re'] == Re]}
            with open(est, 'w', encoding='utf-8') as f:
                json.dump(sorted(keep.values(), key=lambda r: -r['h']), f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
