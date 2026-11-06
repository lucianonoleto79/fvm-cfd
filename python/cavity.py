"""
cavity.py — Solver de Volumes Finitos para escoamento incompressível viscoso 2D
-------------------------------------------------------------------------------
Equações (adimensionais, rho = 1, U = L = 1, nu = 1/Re), regime permanente:
    div(u) = 0
    div(u u) = -grad(p) + (1/Re) lap(u)

Método (mesma formulação de fvm.js):
  - Volumes finitos em malha cartesiana uniforme DESENCONTRADA (Harlow & Welch
    1965): p no centro das células, u nas faces verticais, v nas horizontais.
  - Convecção: upwind implícito + correção diferida (Khosla & Rubin 1974) para
    diferenças centrais -> 2ª ordem no regime convergido (gamma = 1).
  - Difusão: diferenças centrais (2ª ordem).
  - Acoplamento pressão-velocidade: SIMPLEC (Van Doormaal & Raithby 1984).
  - Sistemas lineares: solver esparso direto (SuperLU via scipy) para u, v e p'.

Caso: cavidade com tampa deslizante, [0,1]x[0,1], tampa superior com u = 1.
Validação: Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387-411.

Arranjo dos arrays (índice [i, j], i em x, j em y):
    p : (nx,   ny)    p[i,j] em ((i+.5)dx, (j+.5)dy)
    u : (nx+1, ny)    u[i,j] em (i dx,     (j+.5)dy)
    v : (nx,   ny+1)  v[i,j] em ((i+.5)dx, j dy)

Uso:
    python cavity.py                      # Re = 100, 400, 1000 em 128x128
    python cavity.py --Re 100 --N 64
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import ghia
from monitor import ConvergenceMonitor, add_monitor_args, monitor_from_args

DEFAULT_OUT = Path(__file__).resolve().parent.parent / 'results'


def five_point_matrix(aP, aE, aW, aN, aS):
    """Matriz esparsa de  aP phi_P - aE phi_E - aW phi_W - aN phi_N - aS phi_S,
    com incógnitas ordenadas k = i*nj + j (vizinho N = k+1, vizinho E = k+nj).
    Coeficientes que apontam para fora do domínio devem ser nulos."""
    ni, nj = aP.shape
    n = ni * nj
    diags = [aP.ravel(), -aE.ravel()[:-nj], -aW.ravel()[nj:], -aN.ravel()[:-1], -aS.ravel()[1:]]
    return sp.diags(diags, [0, nj, -nj, 1, -1], shape=(n, n), format='csc')


class CavitySolver:
    def __init__(self, Re=100, N=128, alpha_u=0.8, alpha_p=1.0, gamma=1.0, tol=1e-6):
        if N % 2:
            raise ValueError('N deve ser par (linha central sobre faces)')
        self.Re, self.nx, self.ny = Re, N, N
        self.alpha_u, self.alpha_p, self.gamma, self.tol = alpha_u, alpha_p, gamma, tol
        self.Ulid = 1.0
        self.dx = self.dy = 1.0 / N
        self.u = np.zeros((N + 1, N))
        self.v = np.zeros((N, N + 1))
        self.p = np.zeros((N, N))
        self.dU = np.zeros((N + 1, N))
        self.dV = np.zeros((N, N + 1))
        self.history = []  # (it, ru, rv, rm)
        self.iter = 0
        self.converged = False

    # ------------------------------------------------------------ momento x
    def solve_u(self):
        u, v, p = self.u, self.v, self.p
        dx, dy, a, g = self.dx, self.dy, self.alpha_u, self.gamma
        nu = 1.0 / self.Re
        De, Dn = nu * dy / dx, nu * dx / dy

        uP, uE, uW = u[1:-1], u[2:], u[:-2]                  # faces interiores (nx-1, ny)
        Fe = 0.5 * (uP + uE) * dy
        Fw = 0.5 * (uW + uP) * dy
        Fn = 0.5 * (v[:-1, 1:] + v[1:, 1:]) * dx
        Fs = 0.5 * (v[:-1, :-1] + v[1:, :-1]) * dx
        uN = np.zeros_like(uP); uN[:, :-1] = uP[:, 1:]     # fronteira: fluxo nulo => sem efeito
        uS = np.zeros_like(uP); uS[:, 1:] = uP[:, :-1]

        aE = De + np.maximum(-Fe, 0)
        aW = De + np.maximum(Fw, 0)
        aN = Dn + np.maximum(-Fn, 0)
        aS = Dn + np.maximum(Fs, 0)
        ap0 = aE + aW + aN + aS + (Fe - Fw + Fn - Fs)
        b = (p[:-1] - p[1:]) * dy
        # paredes inferior (u=0) e tampa (u=Ulid): nó de parede a dy/2
        aN[:, -1] = 0; ap0[:, -1] += 2 * Dn - Dn; b[:, -1] += 2 * Dn * self.Ulid
        aS[:, 0] = 0;  ap0[:, 0] += 2 * Dn - Dn
        # correção diferida: F*(phi_CDS - phi_UDS) em cada face
        dc = (Fe * (0.5 * (uP + uE) - np.where(Fe > 0, uP, uE))
              - Fw * (0.5 * (uW + uP) - np.where(Fw > 0, uW, uP))
              + Fn * (0.5 * (uP + uN) - np.where(Fn > 0, uP, uN))
              - Fs * (0.5 * (uS + uP) - np.where(Fs > 0, uS, uP)))
        b -= g * dc
        res = self._residual(ap0, aE, aW, aN, aS, b, uP, uE, uW, uN, uS)
        # sub-relaxação + SIMPLEC
        aPr = ap0 / a
        b = b + (1 - a) * aPr * uP
        sum_nb = aE + aW + aN + aS
        self.dU[1:-1] = dy / np.maximum(aPr - sum_nb, 1e-3 * aPr)
        A = five_point_matrix(aPr, aE, aW, aN, aS)
        u[1:-1] = spla.spsolve(A, b.ravel()).reshape(uP.shape)
        return res

    # ------------------------------------------------------------ momento y
    def solve_v(self):
        u, v, p = self.u, self.v, self.p
        dx, dy, a, g = self.dx, self.dy, self.alpha_u, self.gamma
        nu = 1.0 / self.Re
        De, Dn = nu * dy / dx, nu * dx / dy

        vP, vN, vS = v[:, 1:-1], v[:, 2:], v[:, :-2]          # faces interiores (nx, ny-1)
        Fn = 0.5 * (vP + vN) * dx
        Fs = 0.5 * (vS + vP) * dx
        Fe = 0.5 * (u[1:, :-1] + u[1:, 1:]) * dy
        Fw = 0.5 * (u[:-1, :-1] + u[:-1, 1:]) * dy
        vE = np.zeros_like(vP); vE[:-1] = vP[1:]
        vW = np.zeros_like(vP); vW[1:] = vP[:-1]

        aE = De + np.maximum(-Fe, 0)
        aW = De + np.maximum(Fw, 0)
        aN = Dn + np.maximum(-Fn, 0)
        aS = Dn + np.maximum(Fs, 0)
        ap0 = aE + aW + aN + aS + (Fe - Fw + Fn - Fs)
        b = (p[:, :-1] - p[:, 1:]) * dx
        aE[-1] = 0; ap0[-1] += 2 * De - De                    # paredes laterais (v=0)
        aW[0] = 0;  ap0[0] += 2 * De - De
        # vizinhos N/S na fronteira são nós de Dirichlet com v = 0 (já contados em ap0):
        # retirá-los da matriz evita acoplar ao fim da coluna vizinha na ordenação k = i*nj + j
        aN[:, -1] = 0; aS[:, 0] = 0
        dc = (Fe * (0.5 * (vP + vE) - np.where(Fe > 0, vP, vE))
              - Fw * (0.5 * (vW + vP) - np.where(Fw > 0, vW, vP))
              + Fn * (0.5 * (vP + vN) - np.where(Fn > 0, vP, vN))
              - Fs * (0.5 * (vS + vP) - np.where(Fs > 0, vS, vP)))
        b -= g * dc
        res = self._residual(ap0, aE, aW, aN, aS, b, vP, vE, vW, vN, vS)
        aPr = ap0 / a
        b = b + (1 - a) * aPr * vP
        sum_nb = aE + aW + aN + aS
        self.dV[:, 1:-1] = dx / np.maximum(aPr - sum_nb, 1e-3 * aPr)
        A = five_point_matrix(aPr, aE, aW, aN, aS)
        v[:, 1:-1] = spla.spsolve(A, b.ravel()).reshape(vP.shape)
        return res

    @staticmethod
    def _residual(ap0, aE, aW, aN, aS, b, P, E, W, N, S):
        """Resíduo L1 normalizado da equação não relaxada, avaliado no campo atual."""
        r = ap0 * P - (aE * E + aW * W + aN * N + aS * S) - b
        return np.abs(r).sum() / max(np.abs(ap0 * P).sum(), 1e-30)

    # ------------------------------------------------------ correção de pressão
    def correct_pressure(self):
        u, v, dx, dy = self.u, self.v, self.dx, self.dy
        aE = self.dU[1:] * dy        # dU = 0 nas paredes => coeficiente nulo
        aW = self.dU[:-1] * dy
        aN = self.dV[:, 1:] * dx
        aS = self.dV[:, :-1] * dx
        aP = aE + aW + aN + aS
        b = (u[:-1] - u[1:]) * dy + (v[:, :-1] - v[:, 1:]) * dx
        mass = np.abs(b).sum()
        # Neumann puro: fixa p'(0,0) = 0 para remover a singularidade
        aP, b = aP.copy(), b.copy()
        aP[0, 0] = 1.0; b[0, 0] = 0.0
        aE, aN = aE.copy(), aN.copy()
        aE[0, 0] = aN[0, 0] = 0.0
        A = five_point_matrix(aP, aE, aW, aN, aS)
        pc = spla.spsolve(A, b.ravel()).reshape(aP.shape)
        u[1:-1] += self.dU[1:-1] * (pc[:-1] - pc[1:])
        v[:, 1:-1] += self.dV[:, 1:-1] * (pc[:, :-1] - pc[:, 1:])
        self.p += self.alpha_p * pc
        self.p -= self.p.mean()
        return mass

    def iterate(self):
        ru = self.solve_u()
        rv = self.solve_v()
        rm = self.correct_pressure()
        self.iter += 1
        self.history.append((self.iter, ru, rv, rm))
        if self.iter > 10 and max(ru, rv, rm) < self.tol:
            self.converged = True
        return ru, rv, rm

    def run(self, max_iter=20000, report=100, monitor=None):
        """Itera até convergir. `monitor` (ConvergenceMonitor) imprime os resíduos e
        desenha a curva de convergência; sem ele, só a tabela a cada `report` iterações."""
        if monitor is None:
            monitor = ConvergenceMonitor(['u', 'v', 'massa'], tol=self.tol, report=report)
        t0 = time.perf_counter()
        monitor.start(self)
        while not self.converged and self.iter < max_iter:
            self.iterate()
            monitor.update(self)
        self.seconds = time.perf_counter() - t0
        monitor.finish(self)
        return self.converged

    # ---------------------------------------------------- pós-processamento
    def u_centerline(self):
        y = np.r_[0.0, (np.arange(self.ny) + 0.5) * self.dy, 1.0]
        return y, np.r_[0.0, self.u[self.nx // 2], self.Ulid]

    def v_centerline(self):
        x = np.r_[0.0, (np.arange(self.nx) + 0.5) * self.dx, 1.0]
        return x, np.r_[0.0, self.v[:, self.ny // 2], 0.0]

    def stream_function(self):
        """psi nos vértices: integral de u dy a partir do fundo (psi = 0 nas paredes)."""
        psi = np.zeros((self.nx + 1, self.ny + 1))
        psi[:, 1:] = np.cumsum(self.u, axis=1) * self.dy
        return psi

    def primary_vortex(self):
        """Mínimo de psi com refino parabólico sub-malha."""
        psi = self.stream_function()
        i, j = np.unravel_index(np.argmin(psi), psi.shape)

        def par(fm, f0, fp):
            d = fm - 2 * f0 + fp
            return 0.5 * (fm - fp) / d if d != 0 else 0.0
        sx = par(psi[i - 1, j], psi[i, j], psi[i + 1, j]) if 0 < i < self.nx else 0.0
        sy = par(psi[i, j - 1], psi[i, j], psi[i, j + 1]) if 0 < j < self.ny else 0.0
        return dict(psi=float(psi[i, j]), x=(i + sx) * self.dx, y=(j + sy) * self.dy)

    def compare_ghia(self):
        Re = self.Re
        y, uc = self.u_centerline()
        x, vc = self.v_centerline()
        eu = np.interp(ghia.Y, y, uc) - ghia.U[Re]
        ev = np.interp(ghia.X, x, vc) - ghia.V[Re]
        keep = np.ones_like(ev, bool)
        keep[ghia.SUSPECT_V.get(Re, [])] = False
        vx, gv = self.primary_vortex(), ghia.VORTEX[Re]
        return dict(Re=Re, N=self.nx, iter=self.iter, converged=self.converged,
                    seconds=round(self.seconds, 1),
                    u_rms=float(np.sqrt(np.mean(eu ** 2))), u_max=float(np.abs(eu).max()),
                    v_rms=float(np.sqrt(np.mean(ev ** 2))), v_max=float(np.abs(ev).max()),
                    v_max_clean=float(np.abs(ev[keep]).max()),
                    vortex=vx, ghia_vortex=gv,
                    psi_err_pct=100 * (vx['psi'] - gv['psi']) / abs(gv['psi']))


# -------------------------------------------------------------- gráficos
def plot_case(s, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Re = s.Re
    fig, ax = plt.subplots(2, 2, figsize=(11, 9.5))
    y, uc = s.u_centerline()
    ax[0, 0].plot(uc, y, '-', color='#2f6fd6', lw=2, label=f'FVM {s.nx}×{s.ny}')
    ax[0, 0].plot(ghia.U[Re], ghia.Y, 'o', color='#d9532b', mec='w', label='Ghia et al. (1982)')
    ax[0, 0].set(xlabel='u', ylabel='y', title='u ao longo de x = 0,5')
    x, vc = s.v_centerline()
    ax[0, 1].plot(x, vc, '-', color='#2f6fd6', lw=2, label=f'FVM {s.nx}×{s.ny}')
    ax[0, 1].plot(ghia.X, ghia.V[Re], 'o', color='#d9532b', mec='w', label='Ghia et al. (1982)')
    for k in ghia.SUSPECT_V.get(Re, []):
        ax[0, 1].annotate('erro tipográfico\nem Ghia', (ghia.X[k], ghia.V[Re][k]), xytext=(0.3, -0.3),
                          arrowprops=dict(arrowstyle='->', color='0.4'), fontsize=9, color='0.3')
    ax[0, 1].set(xlabel='x', ylabel='v', title='v ao longo de y = 0,5')
    for a in ax[0]:
        a.grid(alpha=0.3); a.legend(frameon=False)

    psi = s.stream_function()
    xs = np.linspace(0, 1, s.nx + 1)
    lv = np.array(ghia.PSI_LEVELS)
    ax[1, 0].contour(xs, xs, psi.T, levels=lv[lv < 0], colors='#2f6fd6', linewidths=0.8, linestyles='solid')
    ax[1, 0].contour(xs, xs, psi.T, levels=lv[lv > 0], colors='#d9532b', linewidths=0.8, linestyles='solid')
    vx = s.primary_vortex()
    ax[1, 0].plot(vx['x'], vx['y'], 'k.', ms=8)
    ax[1, 0].set(aspect='equal', xlim=(0, 1), ylim=(0, 1), title='Linhas de corrente (níveis de Ghia)')

    h = np.array(s.history)
    ax[1, 1].semilogy(h[:, 0], h[:, 1], color='#2f6fd6', label='u')
    ax[1, 1].semilogy(h[:, 0], h[:, 2], color='#2e8a4e', label='v')
    ax[1, 1].semilogy(h[:, 0], h[:, 3], color='#d9532b', label='massa')
    # na 1ª iteração u = 0 e o resíduo normalizado não tem significado: fixa a escala
    ax[1, 1].set(xlabel='iteração SIMPLEC', ylabel='resíduo normalizado', title='Convergência',
                 ylim=(1e-8, 10))
    ax[1, 1].grid(alpha=0.3, which='both'); ax[1, 1].legend(frameon=False)

    fig.suptitle(f'Cavidade com tampa deslizante — Re = {Re}  (FVM SIMPLEC vs. Ghia et al. 1982)')
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Cavidade com tampa deslizante — FVM/SIMPLEC')
    ap.add_argument('--Re', type=int, nargs='+', default=[100, 400, 1000])
    ap.add_argument('--N', type=int, default=128)
    ap.add_argument('--alpha', type=float, default=0.8)
    ap.add_argument('--tol', type=float, default=1e-6)
    ap.add_argument('--max-iter', type=int, default=20000)
    ap.add_argument('--replot', action='store_true', help='só refaz as figuras a partir dos .npz salvos')
    add_monitor_args(ap, report=100)
    ap.add_argument('--out', default=str(DEFAULT_OUT), help='pasta de saída (padrão: results/ na raiz do repositório)')
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.replot:
        for Re in args.Re:
            d = np.load(OUT / f'cavidade_Re{Re}_N{args.N}.npz')
            s = CavitySolver(Re=Re, N=args.N)
            s.u, s.v, s.p, s.history = d['u'], d['v'], d['p'], d['history'].tolist()
            plot_case(s, OUT / f'cavidade_Re{Re}_N{args.N}.png')
        return

    results = []
    for Re in args.Re:
        print(f'\n=== Re = {Re}, malha {args.N}x{args.N} ===', flush=True)
        s = CavitySolver(Re=Re, N=args.N, alpha_u=args.alpha, tol=args.tol)
        mon = monitor_from_args(args, ['u', 'v', 'massa'], args.tol,
                                OUT / f'convergencia_cavidade_Re{Re}_N{args.N}.png',
                                f'Cavidade Re = {Re}, {args.N}x{args.N}')
        s.run(max_iter=args.max_iter, monitor=mon)
        r = s.compare_ghia()
        results.append(r)
        plot_case(s, OUT / f'cavidade_Re{Re}_N{args.N}.png')
        np.savez(OUT / f'cavidade_Re{Re}_N{args.N}.npz', u=s.u, v=s.v, p=s.p, history=np.array(s.history))

    print('\n Re   N    iter  conv   psi_min    Ghia     dpsi%   centro (x, y)      '
          'u_max   v_max   v_max*  tempo')
    for r in results:
        vx, gv = r['vortex'], r['ghia_vortex']
        print(f"{r['Re']:5d} {r['N']:4d} {r['iter']:6d}  {'sim' if r['converged'] else 'não':4s} "
              f"{vx['psi']:.5f} {gv['psi']:.5f} {r['psi_err_pct']:6.2f}  "
              f"({vx['x']:.4f}, {vx['y']:.4f})  {r['u_max']:.4f}  {r['v_max']:.4f}  "
              f"{r['v_max_clean']:.4f}  {r['seconds']:.0f} s")
    print('* v_max* exclui o ponto suspeito de Ghia (Re = 400, x = 0,9063)')
    with open(OUT / f'resultados_N{args.N}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
