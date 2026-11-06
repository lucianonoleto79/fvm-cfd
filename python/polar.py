"""
Navier–Stokes incompressível 2D, transiente, em malha polar (O) ajustada a um cilindro
circular — volumes finitos em malha desencontrada.

Equações (rho = 1, adimensionais), coordenadas (r, theta):
    div u = (1/r) d(r u_r)/dr + (1/r) du_t/dtheta = 0
    du_r/dt + (1/r) d(r u_r u_r)/dr + (1/r) d(u_t u_r)/dtheta - u_t^2/r
          = -dp/dr + nu [ L(u_r) - u_r/r^2 - (2/r^2) du_t/dtheta ]
    du_t/dt + (1/r^2) d(r^2 u_r u_t)/dr + (1/r) d(u_t u_t)/dtheta
          = -(1/r) dp/dtheta + nu [ L(u_t) - u_t/r^2 + (2/r^2) du_r/dtheta ]
    L(f) = (1/r) d(r df/dr)/dr + (1/r^2) d2f/dtheta2

Malha: theta uniforme e periódica; r_k = R0 exp(k dxi) (células quase quadradas,
Delta r / (r Delta theta) = aspect). Arranjo desencontrado:
    p    (NR, NT)     centros (rc_i, thc_j)
    ur   (NR+1, NT)   faces radiais (rf_i, thc_j); ur[0] = parede, ur[NR] = contorno externo
    ut   (NR, NT)     faces angulares (rc_i, thf_j)

Integração no tempo — passo fracionário (projeção incremental, 2ª ordem):
    convecção (QUICK) e termos cruzados de curvatura: Adams–Bashforth 2 (explícitos)
    difusão "L - 1/r^2": Crank–Nicolson (implícita; matrizes fatoradas uma vez, LU)
    pressão: grad p^n no preditor; Poisson para phi (Neumann, LU fatorada uma vez);
             u^{n+1} = u* - dt grad phi  (divergência nula à precisão de máquina);
             p^{n+1} = p^n + phi
Contorno externo (escoamento livre U = 1 em +x): metade de montante (cos theta < 0)
Dirichlet; metade de jusante, condição convectiva du/dt + U du/dr = 0, com correção
global da vazão para a compatibilidade do Poisson.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from control import run_loop
from monitor import ConvergenceMonitor


# ------------------------------------------------------------------------ malha
class PolarGrid:
    def __init__(self, R0=0.5, Rout=25.0, NT=192, NR=None, aspect=0.7):
        self.R0, self.Rout, self.NT = R0, Rout, NT
        self.dth = 2 * np.pi / NT
        if NR is None:
            NR = int(np.ceil(np.log(Rout / R0) / (aspect * self.dth)))
        self.NR = NR
        self.dxi = np.log(Rout / R0) / NR
        self.rf = R0 * np.exp(self.dxi * np.arange(NR + 1))     # faces radiais
        self.rc = 0.5 * (self.rf[1:] + self.rf[:-1])            # centros
        self.drp = np.diff(self.rf)                             # largura radial das células
        self.drc = np.diff(self.rc)                             # distância entre centros
        self.thf = np.arange(NT) * self.dth                     # faces angulares
        self.thc = self.thf + 0.5 * self.dth
        # distância entre nós de u_t através das faces rf_k (inclui parede e contorno)
        self.dn = np.r_[self.rc[0] - self.rf[0], self.drc, self.rf[-1] - self.rc[-1]]

    @property
    def cells(self):
        return self.NR * self.NT

    def describe(self):
        return (f'{self.NR} x {self.NT} = {self.cells} volumes;  R_ext = {self.Rout / (2 * self.R0):g} D;  '
                f'dr(parede) = {self.drp[0]:.4f} D, r dtheta(parede) = {self.R0 * self.dth:.4f} D, '
                f'dr(externo) = {self.drp[-1]:.3f} D')


# ------------------------------------------------------------- interpolação
def quick_r(a, F):
    """Valores nas faces entre nós consecutivos ao longo do eixo 0 (M nós -> M-1 faces).
    QUICK onde há o nó a montante-montante; central nas bordas."""
    c = 0.5 * (a[:-1] + a[1:])
    qp, qn = c.copy(), c.copy()
    qp[1:] = (6 * a[1:-1] + 3 * a[2:] - a[:-2]) / 8          # F > 0: montante a[k]
    qn[:-1] = (6 * a[1:-1] + 3 * a[:-2] - a[2:]) / 8         # F < 0: montante a[k+1]
    return np.where(F > 0, qp, qn)


def quick_th(a, F):
    """Faces entre j e j+1 (índice j), periódico no eixo 1."""
    a1, am, a2 = np.roll(a, -1, 1), np.roll(a, 1, 1), np.roll(a, -2, 1)
    return np.where(F > 0, (6 * a + 3 * a1 - am) / 8, (6 * a1 + 3 * a - a2) / 8)


def five_point(diag, aS, aN, aT, NT):
    """Matriz (rows*NT) com vizinhos radiais (aS, aN) e angulares periódicos (aT)."""
    rows = diag.shape[0]
    n = rows * NT
    idx = np.arange(n).reshape(rows, NT)
    I, J, V = [idx.ravel()], [idx.ravel()], [diag.ravel()]
    # vizinho interno (i-1) nas linhas i >= 1 e externo (i+1) nas linhas i <= rows-2;
    # os coeficientes que apontam para fora (1ª linha aS, última aN) vão para o contorno
    I += [idx[1:].ravel(), idx[:-1].ravel()]
    J += [idx[:-1].ravel(), idx[1:].ravel()]
    V += [aS[1:].ravel(), aN[:-1].ravel()]
    for s in (1, -1):
        I.append(idx.ravel()); J.append(np.roll(idx, s, 1).ravel()); V.append(aT.ravel())
    return sp.csc_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(n, n))


# ---------------------------------------------------------------------- solver
class PolarNS:
    transient = True

    def __init__(self, grid, Re=100.0, dt=None, cfl=0.4, outer='freestream',
                 wall_speed=None, Uinf=1.0):
        """
        outer       'freestream': escoamento livre Uinf em +x (cilindro)
                    ('wall', ut_outer): parede externa girando (Taylor–Couette)
        wall_speed  função t -> velocidade tangencial da parede interna (rotação)
        """
        g = self.g = grid
        self.Re, self.nu, self.Uinf = Re, Uinf * 2 * g.R0 / Re, Uinf
        self.outer = outer
        self.wall_speed = wall_speed or (lambda t: 0.0)
        NR, NT = g.NR, g.NT
        self.ur = np.zeros((NR + 1, NT))
        self.ut = np.zeros((NR, NT))
        self.p = np.zeros((NR, NT))
        self.ut_b = np.zeros(NT)                     # u_t no contorno externo (rf[NR], thf)
        # metade de jusante; a tolerância evita classificar topo e base (cos = ±1e-16)
        # de forma diferente, o que quebraria a simetria do problema
        self.out = np.cos(g.thc) > 1e-9              # (u_r)
        self.out_t = np.cos(g.thf) > 1e-9            # (u_t)
        self.t = 0.0
        self.iter = 0
        self.history = []                            # (iter, Cd, Cl)
        self.converged = False
        self.prev = None                             # termos explícitos do passo anterior (AB2)
        if outer == 'freestream':
            self.potential_flow()
        else:
            self.ut_b[:] = outer[1]
        if dt is None:
            # CFL do escoamento potencial inicial (|u| chega a 2 Uinf na parede)
            self.dt = 1.0
            dt = cfl / self.cfl()
        self.dt = dt
        self._build()
        if outer == 'freestream':
            self.project()                           # campo inicial discretamente solenoidal

    # --------------------------------------------------------------- matrizes
    def _build(self):
        g, nu, dt = self.g, self.nu, self.dt
        rf, rc, drp, drc, dn, dth, NT = g.rf, g.rc, g.drp, g.drc, g.dn, g.dth, g.NT
        ones = np.ones(NT)
        # L_r - 1/r^2 nos u_r interiores (i = 1..NR-1)
        r = rf[1:-1]
        self.cr_N = (rc[1:] / drp[1:] / drc / r)[:, None] * ones
        self.cr_S = (rc[:-1] / drp[:-1] / drc / r)[:, None] * ones
        self.cr_T = (1 / (r * dth) ** 2)[:, None] * ones
        self.cr_P = -(self.cr_N + self.cr_S + 2 * self.cr_T) - (1 / r ** 2)[:, None]
        # L_t - 1/r^2 nos u_t (i = 0..NR-1); vizinhos de borda: parede e contorno externo
        r = rc
        self.ct_N = (rf[1:] / dn[1:] / drp / r)[:, None] * ones
        self.ct_S = (rf[:-1] / dn[:-1] / drp / r)[:, None] * ones
        self.ct_T = (1 / (r * dth) ** 2)[:, None] * ones
        self.ct_P = -(self.ct_N + self.ct_S + 2 * self.ct_T) - (1 / r ** 2)[:, None]
        h = 0.5 * dt * nu
        Ar = five_point(self.cr_P, self.cr_S, self.cr_N, self.cr_T, NT)
        At = five_point(self.ct_P, self.ct_S, self.ct_N, self.ct_T, NT)
        self.lu_r = spla.splu((sp.identity(Ar.shape[0], format='csc') - h * Ar).tocsc())
        self.lu_t = spla.splu((sp.identity(At.shape[0], format='csc') - h * At).tocsc())
        # Poisson para phi: div(grad phi) em forma de fluxos, Neumann nos contornos
        pN = np.r_[rf[1:-1] / drc, 0.0] * dth
        pS = np.r_[0.0, rf[1:-1] / drc] * dth
        pT = drp / (rc * dth)
        PN, PS, PT = pN[:, None] * ones, pS[:, None] * ones, pT[:, None] * ones
        M = five_point(-(PN + PS + 2 * PT), PS, PN, PT, NT).tolil()
        M[0, :] = 0                                  # fixa o nível de phi (sistema compatível)
        M[0, 0] = 1
        self.lu_p = spla.splu(M.tocsc())

    # ------------------------------------------------------- condição inicial
    def potential_flow(self):
        g, U = self.g, self.Uinf
        a2 = g.R0 ** 2
        R, T = np.meshgrid(g.rf, g.thc, indexing='ij')
        self.ur[...] = U * (1 - a2 / R ** 2) * np.cos(T)
        self.ur[0] = 0.0
        R, T = np.meshgrid(g.rc, g.thf, indexing='ij')
        self.ut[...] = -U * (1 + a2 / R ** 2) * np.sin(T)
        self.ut_b[:] = -U * (1 + a2 / g.Rout ** 2) * np.sin(g.thf)
        R, T = np.meshgrid(g.rc, g.thc, indexing='ij')
        q2 = U ** 2 * (1 - 2 * a2 / R ** 2 * np.cos(2 * T) + a2 ** 2 / R ** 4)
        self.p[...] = 0.5 * (U ** 2 - q2)

    # ------------------------------------------------------ operadores
    def ut_at_rf(self, ut):
        """u_t interpolado linearmente nas faces radiais interiores rf[1..NR-1] (thf)."""
        d = self.g.drp
        return (d[1:, None] * ut[:-1] + d[:-1, None] * ut[1:]) / (d[:-1] + d[1:])[:, None]

    def explicit_terms(self):
        """-(convecção) + nu*(termos cruzados) para u_r interiores e u_t."""
        g, ur, ut = self.g, self.ur, self.ut
        rf, rc, drp, drc, dth = g.rf, g.rc, g.drp, g.drc, g.dth
        uw = self.wall_speed(self.t)
        # --- momento r (linhas i = 1..NR-1)
        Ar = 0.5 * (rf[:-1, None] * ur[:-1] + rf[1:, None] * ur[1:])       # r u_r em rc
        fr = Ar * quick_r(ur, Ar)
        t1 = (fr[1:] - fr[:-1]) / (rf[1:-1] * drc)[:, None]
        Ut = 0.5 * (drp[:-1, None] * ut[:-1] + drp[1:, None] * ut[1:]) / drc[:, None]  # em (rf, thf)
        Fn = np.roll(Ut, -1, 1)                                             # em thf[j+1]
        ft = Fn * quick_th(ur[1:-1], Fn)
        t2 = (ft - np.roll(ft, 1, 1)) / (rf[1:-1, None] * dth)
        utr = self.ut_at_rf(ut)
        utc = 0.5 * (utr + np.roll(utr, -1, 1))
        Cr = t1 + t2 - utc ** 2 / rf[1:-1, None]
        Xr = -2 / rf[1:-1, None] ** 2 * (np.roll(utr, -1, 1) - utr) / dth
        # --- momento theta (linhas i = 0..NR-1)
        Ur = 0.5 * (ur + np.roll(ur, 1, 1))                                 # u_r em (rf, thf)
        qt = np.empty_like(Ur)
        qt[1:-1] = quick_r(ut, Ur[1:-1])
        qt[0], qt[-1] = uw, self.ut_b
        fr = rf[:, None] ** 2 * Ur * qt
        t1 = (fr[1:] - fr[:-1]) / (rc ** 2 * drp)[:, None]
        Uc = 0.5 * (ut + np.roll(ut, -1, 1))                                # em thc
        fc = Uc * quick_th(ut, Uc)
        t2 = (fc - np.roll(fc, 1, 1)) / (rc[:, None] * dth)
        Ct = t1 + t2
        urc = 0.5 * (ur[:-1] + ur[1:])
        Xt = 2 / rc[:, None] ** 2 * (urc - np.roll(urc, 1, 1)) / dth
        return -Cr + self.nu * Xr, -Ct + self.nu * Xt

    def L_r(self, ur):
        """(L - 1/r^2) u_r nos u_r interiores, com os valores de contorno em ur[0], ur[NR]."""
        u = ur[1:-1]
        return (self.cr_P * u + self.cr_N * ur[2:] + self.cr_S * ur[:-2]
                + self.cr_T * (np.roll(u, 1, 1) + np.roll(u, -1, 1)))

    def L_t(self, ut, uw, utb):
        up = np.vstack([ut[1:], utb[None]])
        dn = np.vstack([np.full((1, ut.shape[1]), uw), ut[:-1]])
        return (self.ct_P * ut + self.ct_N * up + self.ct_S * dn
                + self.ct_T * (np.roll(ut, 1, 1) + np.roll(ut, -1, 1)))

    def divergence(self, ur=None, ut=None):
        g = self.g
        ur = self.ur if ur is None else ur
        ut = self.ut if ut is None else ut
        return ((g.rf[1:, None] * ur[1:] - g.rf[:-1, None] * ur[:-1]) * g.dth
                + (np.roll(ut, -1, 1) - ut) * g.drp[:, None])

    def project(self):
        g, dt = self.g, self.dt
        rhs = self.divergence() / dt
        rhs[0, 0] = 0.0
        phi = self.lu_p.solve(rhs.ravel()).reshape(g.NR, g.NT)
        self.ur[1:-1] -= dt * (phi[1:] - phi[:-1]) / g.drc[:, None]
        self.ut -= dt * (phi - np.roll(phi, 1, 1)) / (g.rc[:, None] * g.dth)
        return phi

    # ---------------------------------------------------- contorno externo
    def update_outer(self):
        if self.outer != 'freestream':
            return
        g, dt, U = self.g, self.dt, self.Uinf
        a2 = g.R0 ** 2
        c = U                                         # velocidade convectiva
        # montante: Dirichlet (escoamento potencial no raio externo, ~escoamento livre)
        self.ur[-1, ~self.out] = U * (1 - a2 / g.Rout ** 2) * np.cos(g.thc[~self.out])
        self.ut_b[~self.out_t] = -U * (1 + a2 / g.Rout ** 2) * np.sin(g.thf[~self.out_t])
        # jusante: du/dt + c du/dr = 0
        o, ot = self.out, self.out_t
        self.ur[-1, o] -= dt * c * (self.ur[-1, o] - self.ur[-2, o]) / g.drp[-1]
        self.ut_b[ot] -= dt * c * (self.ut_b[ot] - self.ut[-1, ot]) / g.dn[-1]
        # compatibilidade: vazão líquida nula pelo contorno externo
        q = self.ur[-1]
        qin, qout = q[~o].sum(), q[o].sum()
        if qout > 0:
            self.ur[-1, o] *= -qin / qout

    # ------------------------------------------------------------- passo
    def iterate(self):
        g, dt, nu = self.g, self.dt, self.nu
        h = 0.5 * dt * nu
        Er, Et = self.explicit_terms()
        if self.prev is None:
            self.prev = (Er, Et)
        Pr, Pt = self.prev
        self.prev = (Er, Et)
        Er2, Et2 = 1.5 * Er - 0.5 * Pr, 1.5 * Et - 0.5 * Pt
        uw0 = self.wall_speed(self.t)
        utb0 = self.ut_b.copy()
        ur0b = self.ur.copy()
        Lr0 = self.L_r(ur0b)
        Lt0 = self.L_t(self.ut, uw0, utb0)
        # novos valores de contorno
        self.update_outer()
        uw1 = self.wall_speed(self.t + dt)
        # contribuição de contorno em t^{n+1} (interior nulo)
        z = np.zeros_like(self.ur)
        z[-1] = self.ur[-1]
        br1 = self.L_r(z)
        bt1 = self.L_t(np.zeros_like(self.ut), uw1, self.ut_b)
        gpr = (self.p[1:] - self.p[:-1]) / g.drc[:, None]
        gpt = (self.p - np.roll(self.p, 1, 1)) / (g.rc[:, None] * g.dth)
        rhs_r = ur0b[1:-1] + dt * (Er2 - gpr) + h * Lr0 + h * br1
        rhs_t = self.ut + dt * (Et2 - gpt) + h * Lt0 + h * bt1
        self.ur[1:-1] = self.lu_r.solve(rhs_r.ravel()).reshape(rhs_r.shape)
        self.ut[...] = self.lu_t.solve(rhs_t.ravel()).reshape(rhs_t.shape)
        phi = self.project()
        self.p += phi
        self.t += dt
        self.iter += 1
        cd, cl = self.forces()
        self.history.append((self.iter, cd, cl))
        return cd, cl

    def run(self, n_steps, report=200, monitor=None, control=None):
        if monitor is None:
            monitor = ConvergenceMonitor(['Cd', 'Cl'], report=report, logy=False,
                                         ylabel='coeficiente', dt=self.dt)
        return run_loop(self, n_steps, monitor, control)

    # --------------------------------------------------------------- forças
    def wall_pressure(self):
        g = self.g
        return self.p[0] + (g.R0 - g.rc[0]) * (self.p[1] - self.p[0]) / g.drc[0]

    def wall_shear(self):
        """tau_rtheta = mu (du_t/dr - u_t/r) na parede; derivada por parábola em 3 pontos."""
        g = self.g
        uw = self.wall_speed(self.t)
        x1, x2 = g.rc[0] - g.R0, g.rc[1] - g.R0
        u1, u2 = self.ut[0] - uw, self.ut[1] - uw
        dudr = (u1 * x2 ** 2 - u2 * x1 ** 2) / (x1 * x2 * (x2 - x1))
        return self.nu * (dudr - uw / g.R0)

    def forces(self):
        """Coeficientes de arrasto e sustentação (por unidade de comprimento, D = 2 R0)."""
        g = self.g
        pw, tw = self.wall_pressure(), self.wall_shear()
        ds = g.R0 * g.dth
        fx = (-pw * np.cos(g.thc)).sum() * ds + (-tw * np.sin(g.thf)).sum() * ds
        fy = (-pw * np.sin(g.thc)).sum() * ds + (tw * np.cos(g.thf)).sum() * ds
        q = 0.5 * self.Uinf ** 2 * 2 * g.R0
        return fx / q, fy / q

    def vorticity(self):
        """omega_z nos vértices interiores (rf[1..NR-1], thf)."""
        g = self.g
        rut = g.rc[:, None] * self.ut
        dr = (rut[1:] - rut[:-1]) / g.drc[:, None]
        dth = (self.ur[1:-1] - np.roll(self.ur[1:-1], 1, 1)) / g.dth
        return (dr - dth) / g.rf[1:-1, None]

    def cfl(self):
        g = self.g
        urc = 0.5 * np.abs(self.ur[:-1] + self.ur[1:])
        utc = 0.5 * np.abs(self.ut + np.roll(self.ut, -1, 1))
        return float((self.dt * (urc / g.drp[:, None] + utc / (g.rc[:, None] * g.dth))).max())

    # ------------------------------------------------------- checkpoints
    def get_state(self):
        st = {'ur': self.ur, 'ut': self.ut, 'p': self.p, 'ut_b': self.ut_b,
              't': np.array(self.t), 'dt': np.array(self.dt)}
        if self.prev is not None:
            st['prev_r'], st['prev_t'] = self.prev
        return st

    def set_state(self, st):
        for k in ('ur', 'ut', 'p'):
            if st[k].shape != getattr(self, k).shape:
                raise ValueError(f'checkpoint com malha diferente: {k} {st[k].shape}')
        if abs(float(st['dt']) - self.dt) > 1e-12 * self.dt:
            raise ValueError(f'checkpoint com dt = {float(st["dt"])}, solver com dt = {self.dt}')
        self.ur, self.ut, self.p = (np.array(st[k], dtype=float) for k in ('ur', 'ut', 'p'))
        self.ut_b = np.array(st['ut_b'], dtype=float)
        self.t = float(st['t'])
        self.prev = (np.array(st['prev_r']), np.array(st['prev_t'])) if 'prev_r' in st else None
