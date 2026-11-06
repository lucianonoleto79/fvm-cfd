"""
Navier–Stokes incompressível 2D, transiente, em domínio RETANGULAR com malha cartesiana
esticada e corpo representado por fronteira imersa (ghost-cell) — volumes finitos em
malha desencontrada. Mesmo esquema temporal da malha O (polar.py):
    convecção QUICK + Adams–Bashforth 2; difusão Crank–Nicolson; projeção incremental.

Malha (cartesian_axis): núcleo uniforme de espaçamento h em volta do corpo, refino opcional
na esteira e crescimento geométrico até os contornos. A sequência de faces a partir do núcleo
não depende do tamanho do domínio: domínios maiores só acrescentam células externas — a
malha perto do corpo é a mesma em todos, o que isola o efeito do tamanho do domínio.

Arranjo desencontrado:
    p (NX, NY) centros;  u (NX+1, NY) faces x;  v (NX, NY+1) faces y

Contornos ('freestream'): entrada x = x0 com u = U, v = 0 (Dirichlet); laterais y = y0, y1
com deslizamento livre (v = 0, du/dy = 0); saída x = x1 convectiva (du/dt + U du/dx = 0) com
correção global da vazão. ('closed'): caixa fechada (u = v = 0), usada nos testes.

Fronteira imersa (ghost-cell, Tseng & Ferziger 2003 / Mittal et al. 2008): para cada nó de
velocidade sólido com vizinho fluido (nó fantasma G), toma-se o ponto B mais próximo na
superfície e a imagem I = 2B - G no fluido; u_G = 2 u_B - u_I, com u_I por interpolação
bilinear. Essas linhas entram implicitamente na matriz de momento (fatorada uma vez); nós
sólidos interiores recebem a velocidade de corpo rígido.

Forças: balanço de quantidade de movimento num volume de controle retangular em volta do
corpo (fluxos convectivos, pressão e tensões viscosas nas faces, menos a variação temporal
da quantidade de movimento no interior).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from control import run_loop
from monitor import ConvergenceMonitor
from polar import quick_r


# ------------------------------------------------------------------------ malha
def _grow(start, end, d0, growth, hmax, wake=None):
    """Faces de `start` até `end` (end > start), espaçamento crescendo de d0.
    wake = (x_fim, h_esteira, crescimento): até x_fim o espaçamento não passa de h_esteira."""
    f, d, faces = start, d0, []
    while True:
        if wake is not None and f < wake[0]:
            d = min(d * wake[2], wake[1])
        else:
            d = min(d * growth, hmax)
        f += d
        if f >= end - 1e-12:
            break
        faces.append(f)
    faces.append(end)
    if len(faces) >= 2 and (faces[-1] - faces[-2]) < 0.5 * (faces[-2] - (faces[-3] if len(faces) > 2 else start)):
        faces.pop(-2)                                  # última célula pequena demais: funde
    return faces


def cartesian_axis(lo, hi, core=(-1.5, 1.5), h=0.025, growth=1.05, hmax=1.0, wake=None):
    n = int(round((core[1] - core[0]) / h))
    mid = list(np.linspace(core[0], core[1], n + 1))
    right = _grow(core[1], hi, h, growth, hmax, wake)
    left = [-x for x in _grow(-core[0], -lo, h, growth, hmax)]
    return np.array(left[::-1] + mid + right)


class CartesianGrid:
    def __init__(self, xf, yf):
        self.xf, self.yf = np.asarray(xf, float), np.asarray(yf, float)
        self.NX, self.NY = len(xf) - 1, len(yf) - 1
        self.xc = 0.5 * (self.xf[1:] + self.xf[:-1])
        self.yc = 0.5 * (self.yf[1:] + self.yf[:-1])
        self.dx, self.dy = np.diff(self.xf), np.diff(self.yf)
        self.dxc, self.dyc = np.diff(self.xc), np.diff(self.yc)
        # distâncias entre nós de v em x (inclui contornos) e de u em y
        self.dnx = np.r_[self.xc[0] - self.xf[0], self.dxc, self.xf[-1] - self.xc[-1]]
        self.dny = np.r_[self.yc[0] - self.yf[0], self.dyc, self.yf[-1] - self.yc[-1]]

    @classmethod
    def for_cylinder(cls, Lu=20, Ld=40, H=20, h=0.025, core=1.5, growth=1.05, hmax=1.0,
                     wake=(12.0, 0.1, 1.02)):
        """Domínio [-Lu, Ld] x [-H, H] (em D), cilindro de diâmetro 1 na origem."""
        xf = cartesian_axis(-Lu, Ld, (-core, core), h, growth, hmax, wake)
        yf = cartesian_axis(-H, H, (-core, core), h, growth, hmax)
        g = cls(xf, yf)
        g.Lu, g.Ld, g.H, g.h = Lu, Ld, H, h
        return g

    @property
    def cells(self):
        return self.NX * self.NY

    def describe(self):
        return (f'{self.NX} x {self.NY} = {self.cells} volumes;  domínio [{self.xf[0]:g}, {self.xf[-1]:g}] x '
                f'[{self.yf[0]:g}, {self.yf[-1]:g}] D;  h(núcleo) = {self.dx.min():.4f} D, '
                f'h(máx) = {max(self.dx.max(), self.dy.max()):.3f} D')


# --------------------------------------------------------------------- corpos
class Cylinder:
    """Cilindro de raio R na origem; pode girar com velocidade de superfície speed(t)."""

    def __init__(self, R=0.5, speed=None):
        self.R = R
        self.speed = speed or (lambda t: 0.0)

    def phi(self, x, y):                               # > 0 no fluido
        return np.hypot(x, y) - self.R

    def normal(self, x, y):                            # aponta para o fluido
        r = np.maximum(np.hypot(x, y), 1e-300)
        return x / r, y / r

    def velocity(self, x, y, t):                       # rotação de corpo rígido
        om = self.speed(t) / self.R
        return -om * y, om * x


class Annulus:
    """Fluido entre R0 (girando com velocidade de superfície U0) e R1 (parado): Taylor–Couette."""

    def __init__(self, R0=0.5, R1=1.0, U0=1.0):
        self.R0, self.R1, self.U0 = R0, R1, U0

    def phi(self, x, y):
        r = np.hypot(x, y)
        return np.minimum(r - self.R0, self.R1 - r)

    def normal(self, x, y):
        r = np.maximum(np.hypot(x, y), 1e-300)
        s = np.where(r < 0.5 * (self.R0 + self.R1), 1.0, -1.0)
        return s * x / r, s * y / r

    def velocity(self, x, y, t):
        r = np.hypot(x, y)
        om = np.where(r < 0.5 * (self.R0 + self.R1), self.U0 / self.R0, 0.0)
        return -om * y, om * x


# ------------------------------------------------------------------- matrizes
def five_point(diag, aW, aE, aS, aN):
    """Matriz (nx*ny) com vizinhos W/E (eixo 0) e S/N (eixo 1); coeficientes que apontam
    para fora do bloco (1ª/última linha ou coluna) são ignorados (vão para o contorno)."""
    nx, ny = diag.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    I = [idx.ravel(), idx[1:].ravel(), idx[:-1].ravel(), idx[:, 1:].ravel(), idx[:, :-1].ravel()]
    J = [idx.ravel(), idx[:-1].ravel(), idx[1:].ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel()]
    V = [diag.ravel(), aW[1:].ravel(), aE[:-1].ravel(), aS[:, 1:].ravel(), aN[:, :-1].ravel()]
    n = nx * ny
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(n, n))


def bilinear(X, Y, x, y):
    """Índices (i, j) do canto inferior e pesos dos 4 nós em torno de (x, y)."""
    i = int(np.clip(np.searchsorted(X, x) - 1, 0, len(X) - 2))
    j = int(np.clip(np.searchsorted(Y, y) - 1, 0, len(Y) - 2))
    tx = (x - X[i]) / (X[i + 1] - X[i])
    ty = (y - Y[j]) / (Y[j + 1] - Y[j])
    return [(i, j, (1 - tx) * (1 - ty)), (i + 1, j, tx * (1 - ty)),
            (i, j + 1, (1 - tx) * ty), (i + 1, j + 1, tx * ty)]


# ---------------------------------------------------------------------- solver
class CartesianIBNS:
    transient = True

    def __init__(self, grid, body, Re=100.0, dt=None, cfl=0.35, outer='freestream', Uinf=1.0,
                 box=(-1.25, 1.25, -1.25, 1.25)):
        g = self.g = grid
        self.body, self.Re, self.Uinf, self.outer = body, Re, Uinf, outer
        self.nu = Uinf * 1.0 / Re                     # D = 1
        NX, NY = g.NX, g.NY
        self.u = np.zeros((NX + 1, NY))
        self.v = np.zeros((NX, NY + 1))
        self.p = np.zeros((NX, NY))
        self.vb = np.zeros(NY + 1)                    # v na saída (x = x1)
        if outer == 'freestream':
            self.u[...] = Uinf
        self.dt = dt if dt is not None else cfl * min(g.dx.min(), g.dy.min()) / (2 * Uinf)
        self.t, self.iter, self.history, self.converged, self.prev = 0.0, 0, [], False, None
        # nós (coordenadas) dos componentes
        self.ux, self.uy = g.xf, g.yc
        self.vx, self.vy = g.xc, g.yf
        self._ib_setup()
        self._build()
        # volume de controle das forças (índices de faces); None: sem cálculo de forças
        self.box = None
        if box is not None:
            # faces que contêm a caixa pedida (escolha simétrica numa malha simétrica)
            lo = lambda f, x: int(np.searchsorted(f, x + 1e-9 * max(1.0, abs(x)), side='right') - 1)
            hi = lambda f, x: int(np.searchsorted(f, x - 1e-9 * max(1.0, abs(x)), side='left'))
            ia, ib = lo(g.xf, box[0]), hi(g.xf, box[1])
            ja, jb = lo(g.yf, box[2]), hi(g.yf, box[3])
            if not (1 <= ia < ib <= g.NX - 1 and 1 <= ja < jb <= g.NY - 1):
                raise ValueError('volume de controle das forças fora do domínio')
            self.box = (ia, ib, ja, jb)
        self.M_old = None
        self.apply_ib_explicit()
        self.project()
        if self.box is not None:
            self.M_old = self.momentum_in_box()

    # ------------------------------------------------------- fronteira imersa
    def _classify(self, X, Y, interior_i, interior_j):
        XX, YY = np.meshgrid(X, Y, indexing='ij')
        solid = self.body.phi(XX, YY) <= 0
        ghost = np.zeros_like(solid)
        f = ~solid
        ghost[1:] |= solid[1:] & f[:-1]; ghost[:-1] |= solid[:-1] & f[1:]
        ghost[:, 1:] |= solid[:, 1:] & f[:, :-1]; ghost[:, :-1] |= solid[:, :-1] & f[:, 1:]
        mask = np.zeros_like(solid)
        mask[interior_i, interior_j] = True
        return solid & mask, ghost & mask, XX, YY

    def _ghost_rows(self, X, Y, ghost, off_i, off_j, shape):
        """Para cada nó fantasma: (linha, [(coluna, peso)], ponto B). Índices no bloco interior."""
        rows = []
        ny_int = shape[1]
        for gi, gj in zip(*np.nonzero(ghost)):
            x, y = X[gi], Y[gj]
            ph = self.body.phi(x, y)
            nx, ny = self.body.normal(x, y)
            bx, by = x - ph * nx, y - ph * ny          # ph <= 0: B fica "para fora" de G
            ix, iy = 2 * bx - x, 2 * by - y
            st = []
            for i, j, w in bilinear(X, Y, ix, iy):
                li, lj = i - off_i, j - off_j
                if not (0 <= li < shape[0] and 0 <= lj < shape[1]):
                    raise ValueError('estêncil da imagem fora do bloco interior: corpo perto demais do contorno')
                st.append((li * ny_int + lj, w))
            rows.append(((gi - off_i) * ny_int + (gj - off_j), st, (bx, by)))
        return rows

    def _ib_setup(self):
        g = self.g
        # u: interior i = 1..NX-1, todas as j;  v: todas as i, j = 1..NY-1
        su, gu, UX, UY = self._classify(self.ux, self.uy, slice(1, g.NX), slice(None))
        sv, gv, VX, VY = self._classify(self.vx, self.vy, slice(None), slice(1, g.NY))
        self.u_solid, self.u_ghost = su, gu
        self.v_solid, self.v_ghost = sv, gv
        self.u_rows = self._ghost_rows(self.ux, self.uy, gu, 1, 0, (g.NX - 1, g.NY))
        self.v_rows = self._ghost_rows(self.vx, self.vy, gv, 0, 1, (g.NX, g.NY - 1))
        self.UX, self.UY, self.VX, self.VY = UX, UY, VX, VY

    def _apply_rows(self, A, solid, rows, off):
        """Substitui as linhas dos nós sólidos: fantasmas -> u_G + sum w u_k; interiores -> u."""
        A = A.tolil()
        s = solid[off].ravel()
        for r in np.nonzero(s)[0]:
            A.rows[r] = [r]; A.data[r] = [1.0]
        for r, st, _ in rows:
            acc = {r: 1.0}
            for c, w in st:
                acc[c] = acc.get(c, 0.0) + w
            cols = sorted(acc)
            A.rows[r] = cols; A.data[r] = [acc[c] for c in cols]
        return A.tocsc()

    def ib_rhs(self, comp, t):
        """Lado direito das linhas sólidas (velocidade de corpo rígido / 2 u_B)."""
        if comp == 'u':
            XX, YY, solid, rows, k = self.UX[1:-1], self.UY[1:-1], self.u_solid[1:-1], self.u_rows, 0
        else:
            XX, YY, solid, rows, k = self.VX[:, 1:-1], self.VY[:, 1:-1], self.v_solid[:, 1:-1], self.v_rows, 1
        val = self.body.velocity(XX, YY, t)[k]
        r = np.where(solid, val, 0.0).ravel()
        for row, _, (bx, by) in rows:
            r[row] = 2 * self.body.velocity(np.array(bx), np.array(by), t)[k]
        return r, solid.ravel()

    def apply_ib_explicit(self):
        """Impõe a condição de fronteira imersa no campo atual (usada só na inicialização)."""
        for _ in range(50):                            # iteração simples (linhas acopladas)
            for comp, arr, sl in (('u', self.u, np.s_[1:-1]), ('v', self.v, np.s_[:, 1:-1])):
                blk = arr[sl]
                flat = blk.ravel().copy()
                rhs, solid = self.ib_rhs(comp, self.t)
                flat[solid] = rhs[solid]
                rows = self.u_rows if comp == 'u' else self.v_rows
                for r, st, _ in rows:
                    flat[r] = rhs[r] - sum(w * flat[c] for c, w in st if c != r)
                    w_self = sum(w for c, w in st if c == r)
                    flat[r] /= (1.0 + w_self)
                arr[sl] = flat.reshape(blk.shape)

    # --------------------------------------------------------------- matrizes
    def _build(self):
        g, h = self.g, 0.5 * self.dt * self.nu
        NX, NY = g.NX, g.NY
        # Laplaciano de u nos nós interiores (NX-1, NY)
        cW = (1 / g.dx[:-1] / g.dxc)[:, None] * np.ones(NY)
        cE = (1 / g.dx[1:] / g.dxc)[:, None] * np.ones(NY)
        cS = np.ones((NX - 1, 1)) * np.r_[0.0, 1 / g.dyc / g.dy[1:]][None]
        cN = np.ones((NX - 1, 1)) * np.r_[1 / g.dyc / g.dy[:-1], 0.0][None]
        if self.outer == 'closed':                    # parede: u = 0 em y0, y1 (meia célula)
            cS[:, 0] = 1 / g.dny[0] / g.dy[0]
            cN[:, -1] = 1 / g.dny[-1] / g.dy[-1]
        self.Lu = (cW, cE, cS, cN, -(cW + cE + cS + cN))
        # Laplaciano de v nos nós interiores (NX, NY-1)
        dW = (1 / g.dnx[:-1] / g.dx)[:, None] * np.ones(NY - 1)
        dE = (1 / g.dnx[1:] / g.dx)[:, None] * np.ones(NY - 1)
        dS = np.ones((NX, 1)) * (1 / g.dy[:-1] / g.dyc)[None]
        dN = np.ones((NX, 1)) * (1 / g.dy[1:] / g.dyc)[None]
        self.Lv = (dW, dE, dS, dN, -(dW + dE + dS + dN))
        Au = five_point(*(self.Lu[4], self.Lu[0], self.Lu[1], self.Lu[2], self.Lu[3]))
        Av = five_point(*(self.Lv[4], self.Lv[0], self.Lv[1], self.Lv[2], self.Lv[3]))
        Mu = sp.identity(Au.shape[0], format='csr') - h * Au
        Mv = sp.identity(Av.shape[0], format='csr') - h * Av
        Mu = self._apply_rows(Mu, self.u_solid, self.u_rows, np.s_[1:-1])
        Mv = self._apply_rows(Mv, self.v_solid, self.v_rows, np.s_[:, 1:-1])
        self.lu_u, self.lu_v = spla.splu(Mu), spla.splu(Mv)
        # Poisson (Neumann), fixa uma célula longe do corpo
        PW = np.r_[0.0, 1 / g.dxc][:, None] * g.dy[None]
        PE = np.r_[1 / g.dxc, 0.0][:, None] * g.dy[None]
        PS = g.dx[:, None] * np.r_[0.0, 1 / g.dyc][None]
        PN = g.dx[:, None] * np.r_[1 / g.dyc, 0.0][None]
        M = five_point(-(PW + PE + PS + PN), PW, PE, PS, PN).tolil()
        self.pin = (NX - 1) * NY                       # célula (NX-1, 0): canto da saída
        M.rows[self.pin] = [self.pin]; M.data[self.pin] = [1.0]
        self.lu_p = spla.splu(M.tocsc())

    # -------------------------------------------------------------- operadores
    def L_u(self, u):
        cW, cE, cS, cN, cP = self.Lu
        ui = u[1:-1]
        s = np.zeros_like(ui); n = np.zeros_like(ui)
        s[:, 1:], n[:, :-1] = ui[:, :-1], ui[:, 1:]
        return cP * ui + cW * u[:-2] + cE * u[2:] + cS * s + cN * n

    def L_v(self, v, vin=0.0, vout=None):
        dW, dE, dS, dN, dP = self.Lv
        vi = v[:, 1:-1]
        w = np.vstack([np.full((1, vi.shape[1]), vin), vi[:-1]])
        e = np.vstack([vi[1:], (self.vb if vout is None else vout)[None, 1:-1]])
        return dP * vi + dW * w + dE * e + dS * v[:, :-2] + dN * v[:, 2:]

    def convection(self):
        g, u, v = self.g, self.u, self.v
        # --- momento x nos u interiores
        Fx = 0.5 * (u[:-1] + u[1:])                   # em xc
        fx = Fx * quick_r(u, Fx)
        cx = (fx[1:] - fx[:-1]) / g.dxc[:, None]
        Gy = 0.5 * (g.dx[:-1, None] * v[:-1] + g.dx[1:, None] * v[1:]) / g.dxc[:, None]  # (NX-1, NY+1)
        qy = np.empty_like(Gy)
        qy[:, 1:-1] = quick_r(u[1:-1].T, Gy[:, 1:-1].T).T
        qy[:, 0], qy[:, -1] = u[1:-1, 0], u[1:-1, -1]
        fy = Gy * qy
        cy = (fy[:, 1:] - fy[:, :-1]) / g.dy[None]
        Cu = cx + cy
        # --- momento y nos v interiores
        Fy = 0.5 * (v[:, :-1] + v[:, 1:])             # em yc
        fy = Fy * quick_r(v.T, Fy.T).T
        dy_ = (fy[:, 1:] - fy[:, :-1]) / g.dyc[None]
        Gx = 0.5 * (g.dy[None, :-1] * u[:, :-1] + g.dy[None, 1:] * u[:, 1:]) / g.dyc[None]  # (NX+1, NY-1)
        qx = np.empty_like(Gx)
        qx[1:-1] = quick_r(v[:, 1:-1], Gx[1:-1])
        qx[0], qx[-1] = 0.0, self.vb[1:-1]
        fx = Gx * qx
        dx_ = (fx[1:] - fx[:-1]) / g.dx[:, None]
        Cv = dx_ + dy_
        return -Cu, -Cv

    def divergence(self):
        g = self.g
        return (self.u[1:] - self.u[:-1]) * g.dy[None] + (self.v[:, 1:] - self.v[:, :-1]) * g.dx[:, None]

    def project(self):
        g, dt = self.g, self.dt
        rhs = (self.divergence() / dt).ravel()
        rhs[self.pin] = 0.0
        phi = self.lu_p.solve(rhs).reshape(g.NX, g.NY)
        self.u[1:-1] -= dt * (phi[1:] - phi[:-1]) / g.dxc[:, None]
        self.v[:, 1:-1] -= dt * (phi[:, 1:] - phi[:, :-1]) / g.dyc[None]
        return phi

    def update_outer(self):
        if self.outer != 'freestream':
            return
        g, dt, U = self.g, self.dt, self.Uinf
        self.u[0] = U
        self.u[-1] -= dt * U * (self.u[-1] - self.u[-2]) / g.dx[-1]
        self.vb[1:-1] -= dt * U * (self.vb[1:-1] - self.v[-1, 1:-1]) / g.dnx[-1]
        qin = (self.u[0] * g.dy).sum()
        qout = (self.u[-1] * g.dy).sum()
        self.u[-1] *= qin / qout

    # ------------------------------------------------------------------ passo
    def iterate(self):
        g, dt, nu = self.g, self.dt, self.nu
        h = 0.5 * dt * nu
        Eu, Ev = self.convection()
        if self.prev is None:
            self.prev = (Eu, Ev)
        Pu, Pv = self.prev
        self.prev = (Eu, Ev)
        Eu2, Ev2 = 1.5 * Eu - 0.5 * Pu, 1.5 * Ev - 0.5 * Pv
        Lu0, Lv0 = self.L_u(self.u), self.L_v(self.v)
        u_old0, u_oldN = self.u[0].copy(), self.u[-1].copy()
        self.update_outer()
        # contribuição de contorno em t^{n+1} (interior nulo)
        z = np.zeros_like(self.u); z[0], z[-1] = self.u[0], self.u[-1]
        bu1 = self.L_u(z)
        bv1 = self.L_v(np.zeros_like(self.v))
        gpx = (self.p[1:] - self.p[:-1]) / g.dxc[:, None]
        gpy = (self.p[:, 1:] - self.p[:, :-1]) / g.dyc[None]
        ru = self.u[1:-1] + dt * (Eu2 - gpx) + h * Lu0 + h * bu1
        rv = self.v[:, 1:-1] + dt * (Ev2 - gpy) + h * Lv0 + h * bv1
        ibu, su = self.ib_rhs('u', self.t + dt)
        ibv, sv = self.ib_rhs('v', self.t + dt)
        ru = ru.ravel(); rv = rv.ravel()
        mu = su.copy(); mv = sv.copy()
        for r, _, _ in self.u_rows:
            mu[r] = True
        for r, _, _ in self.v_rows:
            mv[r] = True
        ru[mu] = ibu[mu]; rv[mv] = ibv[mv]
        self.u[1:-1] = self.lu_u.solve(ru).reshape(g.NX - 1, g.NY)
        self.v[:, 1:-1] = self.lu_v.solve(rv).reshape(g.NX, g.NY - 1)
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

    # ------------------------------------------------------------------ forças
    def momentum_in_box(self):
        g, (ia, ib, ja, jb) = self.g, self.box
        uc = 0.5 * (self.u[ia:ib] + self.u[ia + 1:ib + 1])[:, ja:jb]
        vc = 0.5 * (self.v[ia:ib, ja:jb] + self.v[ia:ib, ja + 1:jb + 1])
        A = g.dx[ia:ib, None] * g.dy[None, ja:jb]
        return np.array([(uc * A).sum(), (vc * A).sum()])

    def forces(self):
        """Força do fluido sobre o corpo pelo balanço de quantidade de movimento no VC."""
        if self.box is None:
            return 0.0, 0.0
        g, u, v, p, nu = self.g, self.u, self.v, self.p, self.nu
        ia, ib, ja, jb = self.box
        J = np.arange(ja, jb)
        I = np.arange(ia, ib)
        F = np.zeros(2)
        for i, s in ((ib, 1.0), (ia, -1.0)):           # faces leste (+x) e oeste (-x)
            ue = u[i, J]
            ve = 0.25 * (v[i - 1, J] + v[i - 1, J + 1] + v[i, J] + v[i, J + 1])
            pe = 0.5 * (p[i - 1, J] + p[i, J])
            txx = 2 * nu * (u[i + 1, J] - u[i - 1, J]) / (g.xf[i + 1] - g.xf[i - 1])
            dudy = (u[i, J + 1] - u[i, J - 1]) / (g.yc[J + 1] - g.yc[J - 1])
            vcy = 0.5 * (v[:, J] + v[:, J + 1])
            dvdx = (vcy[i] - vcy[i - 1]) / (g.xc[i] - g.xc[i - 1])
            txy = nu * (dudy + dvdx)
            dy = g.dy[J]
            F[0] += s * ((-pe + txx - ue * ue) * dy).sum()
            F[1] += s * ((txy - ve * ue) * dy).sum()
        for j, s in ((jb, 1.0), (ja, -1.0)):           # faces norte (+y) e sul (-y)
            vn = v[I, j]
            un = 0.25 * (u[I, j - 1] + u[I + 1, j - 1] + u[I, j] + u[I + 1, j])
            pn = 0.5 * (p[I, j - 1] + p[I, j])
            tyy = 2 * nu * (v[I, j + 1] - v[I, j - 1]) / (g.yf[j + 1] - g.yf[j - 1])
            ucx = 0.5 * (u[I] + u[I + 1])
            dudy = (ucx[:, j] - ucx[:, j - 1]) / (g.yc[j] - g.yc[j - 1])
            dvdx = (v[I + 1, j] - v[I - 1, j]) / (g.xc[I + 1] - g.xc[I - 1])
            txy = nu * (dudy + dvdx)
            dx = g.dx[I]
            F[0] += s * ((txy - un * vn) * dx).sum()
            F[1] += s * ((-pn + tyy - vn * vn) * dx).sum()
        M = self.momentum_in_box()
        if self.M_old is not None:
            F -= (M - self.M_old) / self.dt
        self.M_old = M
        q = 0.5 * self.Uinf ** 2 * 1.0
        return F[0] / q, F[1] / q

    def vorticity(self):
        """omega_z nos vértices interiores (xf[1..NX-1], yf[1..NY-1])."""
        g = self.g
        dvdx = (self.v[1:, 1:-1] - self.v[:-1, 1:-1]) / g.dxc[:, None]
        dudy = (self.u[1:-1, 1:] - self.u[1:-1, :-1]) / g.dyc[None]
        return dvdx - dudy

    def cfl(self):
        g = self.g
        uc = 0.5 * np.abs(self.u[:-1] + self.u[1:])
        vc = 0.5 * np.abs(self.v[:, :-1] + self.v[:, 1:])
        return float((self.dt * (uc / g.dx[:, None] + vc / g.dy[None])).max())

    # ------------------------------------------------------------ checkpoints
    def get_state(self):
        st = {'u': self.u, 'v': self.v, 'p': self.p, 'vb': self.vb, 't': np.array(self.t),
              'dt': np.array(self.dt), 'M_old': np.zeros(2) if self.M_old is None else self.M_old}
        if self.prev is not None:
            st['prev_u'], st['prev_v'] = self.prev
        return st

    def set_state(self, st):
        for k in ('u', 'v', 'p'):
            if st[k].shape != getattr(self, k).shape:
                raise ValueError(f'checkpoint com malha diferente: {k} {st[k].shape}')
        if abs(float(st['dt']) - self.dt) > 1e-12 * self.dt:
            raise ValueError(f'checkpoint com dt = {float(st["dt"])}, solver com dt = {self.dt}')
        self.u, self.v, self.p = (np.array(st[k], dtype=float) for k in ('u', 'v', 'p'))
        self.vb, self.t = np.array(st['vb'], dtype=float), float(st['t'])
        self.M_old = np.array(st['M_old'], dtype=float)
        self.prev = (np.array(st['prev_u']), np.array(st['prev_v'])) if 'prev_u' in st else None
