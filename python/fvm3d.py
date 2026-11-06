"""
fvm3d.py — Solver 3D de Volumes Finitos para escoamento incompressível viscoso
-------------------------------------------------------------------------------
Equações (regime permanente, propriedades constantes):
    div(u) = 0
    rho div(u u) = -grad(p) + mu lap(u)

Extensão tridimensional de cavity.py (mesma formulação):
  - Malha cartesiana uniforme DESENCONTRADA: p no centro das células,
    u, v, w nas faces normais a x, y, z (Harlow & Welch 1965).
  - Convecção: upwind implícito + correção diferida para CDS (2ª ordem).
  - Difusão: diferenças centrais.
  - Acoplamento pressão-velocidade: SIMPLEC (Van Doormaal & Raithby 1984).
  - Geometrias não cartesianas por FRONTEIRA IMERSA: uma função level set
    phi(x,y,z) (< 0 no fluido) marca as células fluidas. A continuidade é
    imposta nas células fluidas (faces fluido-sólido fechadas); nos termos
    difusivos do momento, o vizinho sólido é substituído pela parede real, à
    distância theta*h obtida por interpolação de phi (tipo Shortley-Weller),
    o que mantém a precisão de 2ª ordem da velocidade junto à parede curva.
  - Sistemas lineares: BiCGSTAB (pré-condicionador de Jacobi) para o momento;
    Gradiente Conjugado com multigrid algébrico (pyamg, se disponível) para a
    correção de pressão.

Condições de contorno nas 6 faces do domínio, chaves (eixo, lado), lado = -1/+1:
    ('dirichlet', (u, v, w))  parede (velocidade dada) ou entrada
    ('outlet',)               saída: gradiente nulo, vazão corrigida para
                              conservar massa globalmente (só em z+)

Arranjo (índices [i, j, k]):
    p   : (nx, ny, nz)       centros
    V[0]: (nx+1, ny, nz)     u nas faces x
    V[1]: (nx, ny+1, nz)     v nas faces y
    V[2]: (nx, ny, nz+1)     w nas faces z
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from control import run_loop
from monitor import ConvergenceMonitor

try:
    import pyamg
except ImportError:  # pragma: no cover
    pyamg = None


def take(a, axis, start, stop):
    """Fatia a[start:stop] ao longo de axis (vista, sem cópia)."""
    idx = [slice(None)] * a.ndim
    idx[axis] = slice(start, stop)
    return a[tuple(idx)]


def shift(a, axis, s, fill):
    """b[i] = a[i+s] ao longo de axis (s = +-1); posições fora do domínio recebem fill."""
    out = np.empty_like(a)
    n = a.shape[axis]
    if s > 0:
        take(out, axis, 0, n - 1)[...] = take(a, axis, 1, n)
        take(out, axis, n - 1, n)[...] = fill
    else:
        take(out, axis, 1, n)[...] = take(a, axis, 0, n - 1)
        take(out, axis, 0, 1)[...] = fill
    return out


def edge_mask(shape, axis, s):
    """Máscara da última (s=+1) ou primeira (s=-1) camada ao longo de axis."""
    m = np.zeros(shape, bool)
    n = shape[axis]
    take(m, axis, n - 1, n)[...] = s > 0
    take(m, axis, 0, 1)[...] = s < 0
    return m


class Grid3D:
    def __init__(self, n, lo, hi):
        self.n = tuple(int(v) for v in n)
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.h = (self.hi - self.lo) / np.array(self.n)
        self.area = np.array([self.h[1] * self.h[2], self.h[0] * self.h[2], self.h[0] * self.h[1]])

    def centers(self, a):
        return self.lo[a] + (np.arange(self.n[a]) + 0.5) * self.h[a]

    def faces(self, a):
        return self.lo[a] + np.arange(self.n[a] + 1) * self.h[a]

    def coords(self, c=None):
        """Coordenadas (X, Y, Z) dos nós da componente c (None = centros das células)."""
        ax = [self.faces(a) if a == c else self.centers(a) for a in range(3)]
        return np.meshgrid(*ax, indexing='ij')


class _Neighbor:
    """Classificação estática de um vizinho (componente c, direção a, lado s).
    kind: 0 = incógnita ativa, 1 = valor conhecido (Dirichlet), 2 = gradiente nulo."""
    __slots__ = ('kind', 'dist', 'nidx', 'ghost', 'reg')


class SIMPLEC3D:
    def __init__(self, grid, mu, rho=1.0, phi=None, bc=None, alpha_u=0.8, alpha_p=1.0,
                 gamma=1.0, Uref=1.0, tol=1e-6, p_tol=1e-3, u_tol=1e-8, amg_every=20):
        self.g, self.mu, self.rho = grid, mu, rho
        self.u_tol = u_tol
        self.alpha_u, self.alpha_p, self.gamma = alpha_u, alpha_p, gamma
        self.Uref, self.tol, self.p_tol, self.amg_every = Uref, tol, p_tol, amg_every
        self.bc = {(a, s): ('dirichlet', (0.0, 0.0, 0.0)) for a in range(3) for s in (-1, 1)}
        self.bc.update(bc or {})
        for key, val in self.bc.items():
            if val[0] == 'outlet' and key != (2, 1):
                raise NotImplementedError('saída implementada apenas em z+')
        n = grid.n
        self.phi = phi
        self.fluid = phi(*grid.coords()) < 0 if phi else np.ones(n, bool)
        self.p = np.zeros(n)
        self.V, self.act, self.idx, self.d, self.nb = [], [], [], [], []
        for c in range(3):
            shape = tuple(n[a] + (a == c) for a in range(3))
            V = np.zeros(shape)
            act = np.zeros(shape, bool)
            take(act, c, 1, n[c])[...] = take(self.fluid, c, 0, n[c] - 1) & take(self.fluid, c, 1, n[c])
            idx = -np.ones(shape, int)
            idx[act] = np.arange(act.sum())
            # valores prescritos nas faces de contorno normais a c (entrada / parede)
            for s, (i0, i1) in ((-1, (0, 1)), (1, (n[c], n[c] + 1))):
                kind = self.bc[(c, s)]
                if kind[0] == 'dirichlet':
                    cell = take(self.fluid, c, 0, 1) if s < 0 else take(self.fluid, c, n[c] - 1, n[c])
                    take(V, c, i0, i1)[...] = kind[1][c] * cell
            self.V.append(V); self.act.append(act); self.idx.append(idx)
            self.d.append(np.zeros(shape))
        for c in range(3):
            self.nb.append(self._classify(c))
        self.cell_idx = None
        self.history = []
        self.iter = 0
        self.converged = False
        self._amg = None

    # ---------------------------------------------------------- geometria
    def _classify(self, c):
        g, n = self.g, self.g.n
        act_s = take(self.act[c], c, 1, n[c])
        idx_s = take(self.idx[c], c, 1, n[c])
        if self.phi:
            phi_full = self.phi(*g.coords(c))
            phiP = take(phi_full, c, 1, n[c])
        out = {}
        for a in range(3):
            for s in (-1, 1):
                nb = _Neighbor()
                if a == c:
                    lo = 2 if s > 0 else 0
                    nact = take(self.act[c], c, lo, lo + n[c] - 1)
                    nb.nidx = take(self.idx[c], c, lo, lo + n[c] - 1)
                    boundary = edge_mask(act_s.shape, c, s)   # vizinho é nó de contorno
                    oob = np.zeros_like(boundary)
                    phiN = take(phi_full, c, lo, lo + n[c] - 1) if self.phi else None
                    nb.ghost = 0.0
                else:
                    nact = shift(act_s, a, s, False)
                    nb.nidx = shift(idx_s, a, s, -1)
                    oob = edge_mask(act_s.shape, a, s)        # vizinho fora do domínio
                    boundary = np.zeros_like(oob)
                    phiN = shift(phiP, a, s, 1.0) if self.phi else None
                    bc = self.bc[(a, s)]
                    nb.ghost = bc[1][c] if bc[0] == 'dirichlet' else 0.0
                kind = np.where(nact, 0, 1)
                dist = np.ones(act_s.shape)
                solid = ~nact & ~boundary & ~oob
                if self.phi:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        theta = np.clip(phiP / (phiP - phiN), 0.05, 1.0)
                    theta = np.where(phiN > 0, theta, 1.0)
                    dist[solid] = theta[solid]
                if a != c:
                    if self.bc[(a, s)][0] == 'outlet':
                        kind[oob] = 2
                    else:
                        dist[oob] = 0.5                       # parede/entrada a h/2
                nb.kind, nb.dist = kind, dist
                nb.reg = (kind == 0) | ((kind == 1) & (dist == 1.0))  # faces com CDS válido
                out[(a, s)] = nb
        return out

    # ---------------------------------------------------------- fluxos
    def _flux(self, c, a, s):
        """Fluxo de massa (sentido +a) na face (a, s) dos VCs interiores de c."""
        n, V, A = self.g.n, self.V, self.g.area[a] * self.rho
        if a == c:
            lo = 1 if s > 0 else 0
            return 0.5 * (take(V[c], c, lo, lo + n[c] - 1) + take(V[c], c, lo + 1, lo + n[c])) * A
        Va = 0.5 * (take(V[a], c, 0, n[c] - 1) + take(V[a], c, 1, n[c]))
        return (take(Va, a, 1, n[a] + 1) if s > 0 else take(Va, a, 0, n[a])) * A

    # ---------------------------------------------------------- momento
    def momentum(self, c):
        g, n, V = self.g, self.g.n, self.V
        P = take(V[c], c, 1, n[c])
        act = take(self.act[c], c, 1, n[c])
        idx = take(self.idx[c], c, 1, n[c])
        ap0 = np.zeros(P.shape); b = np.zeros(P.shape); dc = np.zeros(P.shape)
        sum_act = np.zeros(P.shape); off = np.zeros(P.shape)
        rows, cols, vals = [], [], []
        for (a, s), nb in self.nb[c].items():
            D = self.mu * g.area[a] / g.h[a]
            Fo = s * self._flux(c, a, s)                       # fluxo para fora do VC
            if a == c:
                lo = 2 if s > 0 else 0
                N = take(V[c], c, lo, lo + n[c] - 1)
            else:
                N = shift(P, a, s, nb.ghost)
            coef = D / nb.dist + np.maximum(-Fo, 0.0)
            coef[nb.kind == 2] = 0.0
            ap0 += coef + Fo
            up = np.where(Fo > 0, P, N)
            dc += np.where(nb.reg, Fo * (0.5 * (P + N) - up), 0.0)
            known = nb.kind == 1
            b += np.where(known, coef * N, 0.0)
            unk = nb.kind == 0
            off += np.where(unk, coef * N, 0.0)
            sum_act += np.where(unk, coef, 0.0)
            m = unk & act
            rows.append(idx[m]); cols.append(nb.nidx[m]); vals.append(-coef[m])
        b += (take(self.p, c, 0, n[c] - 1) - take(self.p, c, 1, n[c])) * g.area[c] - self.gamma * dc
        # resíduo da equação não relaxada, normalizado por sum|aP| * Uref
        res = np.abs((ap0 * P - off - b)[act]).sum() / (np.abs(ap0[act]).sum() * self.Uref + 1e-30)
        aPr = ap0 / self.alpha_u
        b += (1 - self.alpha_u) * aPr * P
        d = take(self.d[c], c, 1, n[c])
        d[...] = np.where(act, g.area[c] / np.maximum(aPr - sum_act, 1e-3 * aPr), 0.0)
        nu = int(act.sum())
        A = sp.csr_matrix((np.concatenate(vals + [aPr[act]]),
                           (np.concatenate(rows + [idx[act]]), np.concatenate(cols + [idx[act]]))),
                          shape=(nu, nu))
        M = sp.diags(1.0 / aPr[act])
        # tolerância relativa a ||b||, que contém termos grandes (pressão, contornos):
        # precisa ser bem menor que a tolerância de convergência externa
        x, _ = spla.bicgstab(A, b[act], x0=P[act], rtol=self.u_tol, maxiter=500, M=M)
        P[act] = x
        return res

    # ---------------------------------------------------------- saída
    def inflow(self):
        """Vazão mássica entrando pelas faces Dirichlet do domínio."""
        n, A, Q = self.g.n, self.g.area, 0.0
        for a in range(3):
            if self.bc[(a, -1)][0] == 'dirichlet':
                Q += take(self.V[a], a, 0, 1).sum() * A[a]
            if self.bc[(a, 1)][0] == 'dirichlet':
                Q -= take(self.V[a], a, n[a], n[a] + 1).sum() * A[a]
        return self.rho * Q

    def apply_outlet(self):
        if self.bc[(2, 1)][0] != 'outlet':
            return
        w, fl = self.V[2], self.fluid[:, :, -1]
        w[:, :, -1] = np.where(fl, w[:, :, -2], 0.0)
        Qout = self.rho * w[:, :, -1].sum() * self.g.area[2]
        if Qout > 0:
            w[:, :, -1] *= self.inflow() / Qout

    # ---------------------------------------------------------- pressão
    def pressure_correction(self):
        g, n, V, d = self.g, self.g.n, self.V, self.d
        aP = np.zeros(n); b = np.zeros(n); hi = []
        for a in range(3):
            A = g.area[a] * self.rho
            aP += (take(d[a], a, 0, n[a]) + take(d[a], a, 1, n[a] + 1)) * A
            b += (take(V[a], a, 0, n[a]) - take(V[a], a, 1, n[a] + 1)) * A
            hi.append(take(d[a], a, 1, n[a]) * A)
        mass = np.abs(b[self.fluid]).sum()
        if self.cell_idx is None:
            # incógnitas: células fluidas, exceto uma de referência (p' = 0): Neumann puro
            unk = self.fluid.copy()
            unk.flat[np.flatnonzero(unk)[0]] = False
            self.cell_unk = unk
            self.cell_idx = -np.ones(n, int)
            self.cell_idx[unk] = np.arange(unk.sum())
        unk, cid = self.cell_unk, self.cell_idx
        rows, cols, vals = [cid[unk]], [cid[unk]], [aP[unk]]
        for a in range(3):
            i0, i1 = take(cid, a, 0, n[a] - 1), take(cid, a, 1, n[a])
            m = (i0 >= 0) & (i1 >= 0)
            v = -hi[a][m]
            rows += [i0[m], i1[m]]; cols += [i1[m], i0[m]]; vals += [v, v]
        N = int(unk.sum())
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
        if pyamg is not None:
            if self._amg is None or self.iter % self.amg_every == 0:
                self._amg = pyamg.smoothed_aggregation_solver(A, symmetry='hermitian').aspreconditioner()
            M = self._amg
        else:
            M = sp.diags(1.0 / A.diagonal())
        x, _ = spla.cg(A, b[unk], rtol=self.p_tol, maxiter=500, M=M)
        pc = np.zeros(n)
        pc[unk] = x
        for a in range(3):
            take(V[a], a, 1, n[a])[...] += take(d[a], a, 1, n[a]) * (take(pc, a, 0, n[a] - 1) - take(pc, a, 1, n[a]))
        self.p += self.alpha_p * pc
        return mass

    # ---------------------------------------------------------- laço
    def iterate(self):
        r = [self.momentum(c) for c in range(3)]
        self.apply_outlet()
        Qref = self.inflow() or self.rho * self.Uref * self.g.area.max()
        rm = self.pressure_correction() / abs(Qref)
        self.iter += 1
        self.history.append((self.iter, *r, rm))
        if self.iter > 10 and max(*r, rm) < self.tol:
            self.converged = True
        return (*r, rm)

    def run(self, max_iter=5000, report=50, monitor=None, control=None):
        """Itera até convergir. `monitor` (ConvergenceMonitor) imprime os resíduos e
        desenha a curva de convergência; sem ele, só a tabela a cada `report` iterações.
        `control` (RunControl) permite pausar, salvar parciais e interromper."""
        if monitor is None:
            monitor = ConvergenceMonitor(['u', 'v', 'w', 'massa'], tol=self.tol, report=report)
        return run_loop(self, max_iter, monitor, control)

    def get_state(self):
        return {'u': self.V[0], 'v': self.V[1], 'w': self.V[2], 'p': self.p}

    def set_state(self, st):
        new = [st['u'], st['v'], st['w'], st['p']]
        old = [*self.V, self.p]
        for a, b in zip(new, old):
            if a.shape != b.shape:
                raise ValueError(f'checkpoint com malha diferente: {a.shape} != {b.shape}')
        self.V = [np.array(a, dtype=float) for a in new[:3]]
        self.p = np.array(new[3], dtype=float)
        self._amg = None                             # hierarquia AMG é refeita
