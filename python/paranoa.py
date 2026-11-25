"""
Lago Paranoá (DF): dispersão de um poluente como escalar passivo — o caso `paranoa` do
Cordel (https://github.com/lucianonoleto79/cordel, casos/paranoa.py) resolvido com volumes
finitos em malha cartesiana desencontrada.

Modelo (o mesmo do Cordel): Navier–Stokes incompressível 2D no plano horizontal (médio na
vertical, profundidade uniforme H, sem superfície livre) + advecção-difusão de C/C0:
    du/dt + (u.grad)u = -grad p + nu lap u,   div u = 0
    dC/dt + u.grad C  = kappa lap C
  - geometria: contorno do OpenStreetMap usado pelo Cordel (paranoa_contorno.json, UTM 23S,
    © OpenStreetMap contributors, ODbL), rasterizado: são fluidas as células cujo centro está
    dentro do polígono (contorno em "escada");
  - entradas: fozes dos cinco afluentes; velocidade normal uniforme nas faces de contorno a
    até R_ENTRADA da foz, ajustada para a vazão Q/H (por unidade de profundidade) ser exata;
  - saída: barragem, vazão prescrita igual à soma das entradas (faces a até R_SAIDA);
  - margens: u = v = 0 (velocidade normal nula na face; tangencial por célula fantasma);
  - escalar: C = 1 (Dirichlet, advecção + difusão pela meia célula) nas faces da foz do
    Riacho Fundo, C = 0 nas demais entradas, fluxo nulo nas margens, só advecção na saída.

Numérica (a mesma família de cartesian.py / polar.py):
  convecção Adams–Bashforth 2 (central), difusão Crank–Nicolson, projeção incremental com
  Poisson fatorado uma vez (splu); escalar explícito em volumes finitos com limitador TVD de
  van Leer (sem concentrações negativas nem acima de 1), com subpassos pelo CFL.

Parâmetros ILUSTRATIVOS, não calibrados (ver o Cordel): vazões, H, nu e kappa são ordens de
grandeza; sem vento, Coriolis, atrito de fundo, batimetria nem estratificação.

    python python/paranoa.py --dias 7                      (1 semana, h = 80 m, dt = 1800 s)
    python python/paranoa.py --dias 7 --h 40 --dt 900      (malha mais fina)
    python python/paranoa.py --apenas-malha                (só domínio e malha)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage

from casefiles import CaseDir
from control import run_loop
from monitor import ConvergenceMonitor

REPO = Path(__file__).resolve().parent.parent
CONTORNO = REPO / 'python' / 'dados' / 'paranoa_contorno.json'

H = 12.0                    # profundidade uniforme (m)
VAZOES = dict(bananal=1.5, torto=2.0, riacho_fundo=2.5, gama=2.0, cabeca_de_veado=0.5)  # m³/s
FONTE = 'riacho_fundo'
NU, KAPPA = 0.1, 0.1        # viscosidade e difusividade horizontais (m²/s)
R_ENTRADA, R_SAIDA = 150.0, 250.0
DIA = 86400.0
RAMPA = 1.0 * DIA
# nós das sondas do Cordel (no_central de casos/paranoa.py com a malha de 80 m)
SONDAS = {'Braço do Riacho Fundo': (190037.6, 8246848.2), 'Braço Sul': (194857.4, 8247468.7),
          'Corpo central': (197561.2, 8250827.5), 'Braço Norte': (195427.9, 8255268.4),
          'Barragem': (200550.9, 8251932.0)}


def rampa(t):
    return 0.5 * (1 - np.cos(np.pi * min(t / RAMPA, 1.0)))


# ------------------------------------------------------------------------ malha
class LagoGrid:
    """Malha cartesiana uniforme de passo h com a máscara do lago."""

    def __init__(self, h=80.0, contorno=CONTORNO):
        from matplotlib.path import Path as MPath
        D = json.load(open(contorno))
        self.anel = np.array(D['contorno'], float)
        self.entradas = {k: np.array(v) for k, v in D['entradas'].items()}
        self.saida = np.array(D['saida'])
        self.h = h
        lo, hi = self.anel.min(0) - 2 * h, self.anel.max(0) + 2 * h
        self.NX, self.NY = (np.ceil((hi - lo) / h)).astype(int)
        self.x0, self.y0 = lo
        self.xc = self.x0 + (np.arange(self.NX) + 0.5) * h
        self.yc = self.y0 + (np.arange(self.NY) + 0.5) * h
        XX, YY = np.meshgrid(self.xc, self.yc, indexing='ij')
        wet = MPath(self.anel).contains_points(np.column_stack([XX.ravel(), YY.ravel()]))
        wet = wet.reshape(self.NX, self.NY)
        lab, n = ndimage.label(wet)                    # só o maior conjunto conexo (vizinhança 4)
        if n > 1:
            wet = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        self.wet = wet
        self.XX, self.YY = XX, YY

    def celula(self, p):
        """Célula molhada mais próxima do ponto p."""
        d = np.hypot(self.XX - p[0], self.YY - p[1])
        d[~self.wet] = np.inf
        return np.unravel_index(np.argmin(d), d.shape)


# ---------------------------------------------------------------------- solver
class ParanoaFVM:
    transient = True

    def __init__(self, g, dt=1800.0, nu=NU, kappa=KAPPA):
        self.g, self.dt, self.nu, self.kappa = g, dt, nu, kappa
        NX, NY, h, w = g.NX, g.NY, g.h, g.wet
        self.u = np.zeros((NX + 1, NY))
        self.v = np.zeros((NX, NY + 1))
        self.p = np.zeros((NX, NY))
        self.C = np.zeros((NX, NY))
        self.t, self.iter, self.converged, self.history, self.prev = 0.0, 0, False, [], None

        # faces abertas (fluido dos dois lados) e de contorno (fluido de um lado só)
        wl = np.zeros((NX + 1, NY), bool); wr = np.zeros_like(wl)
        wl[1:], wr[:-1] = w, w
        self.uo, self.ub = wl & wr, wl ^ wr
        self.u_out = np.where(wl & ~wr, 1.0, np.where(wr & ~wl, -1.0, 0.0))   # normal externa (x)
        ws = np.zeros((NX, NY + 1), bool); wn = np.zeros_like(ws)
        ws[:, 1:], wn[:, :-1] = w, w
        self.vo, self.vb = ws & wn, ws ^ wn
        self.v_out = np.where(ws & ~wn, 1.0, np.where(wn & ~ws, -1.0, 0.0))

        # aberturas: faces de contorno a até R da face mais próxima da foz / barragem
        bu = np.argwhere(self.ub); bv = np.argwhere(self.vb)
        cu = np.column_stack([g.x0 + bu[:, 0] * h, g.yc[bu[:, 1]]])
        cv = np.column_stack([g.xc[bv[:, 0]], g.y0 + bv[:, 1] * h])
        cent = np.vstack([cu, cv])

        def faces(p, r):
            k0 = np.argmin(np.hypot(*(cent - p).T))
            sel = np.hypot(*(cent - cent[k0]).T) <= r
            return bu[sel[:len(bu)]], bv[sel[len(bu):]], float(np.hypot(*(cent[k0] - p)))

        self.aberturas = {}
        for nome, Q in VAZOES.items():
            fu, fv, d = faces(g.entradas[nome], R_ENTRADA)
            self.aberturas[nome] = dict(fu=fu, fv=fv, Q=Q, U=Q / H / (h * (len(fu) + len(fv))),
                                        C=1.0 if nome == FONTE else 0.0, sinal=-1.0, dist=d)
        fu, fv, d = faces(g.saida, R_SAIDA)
        Qs = sum(VAZOES.values())
        self.aberturas['barragem'] = dict(fu=fu, fv=fv, Q=Qs, U=Qs / H / (h * (len(fu) + len(fv))),
                                          C=None, sinal=1.0, dist=d)
        # velocidades nominais nas faces de contorno (0 nas margens) e C imposto nas entradas
        self.ubn, self.vbn = np.zeros_like(self.u), np.zeros_like(self.v)
        self.Cin_u, self.Cin_v = np.full(self.u.shape, np.nan), np.full(self.v.shape, np.nan)
        for a in self.aberturas.values():
            for (i, j) in a['fu']:
                self.ubn[i, j] = a['sinal'] * a['U'] * self.u_out[i, j]
                if a['C'] is not None:
                    self.Cin_u[i, j] = a['C']
            for (i, j) in a['fv']:
                self.vbn[i, j] = a['sinal'] * a['U'] * self.v_out[i, j]
                if a['C'] is not None:
                    self.Cin_v[i, j] = a['C']
        self.sondas = {k: g.celula(p) for k, p in SONDAS.items()}
        self.snaps = {}
        self.balanco = dict(entrada_adv=0.0, entrada_dif=0.0, saida=0.0)   # ∫ fluxo de C (m²)
        self._build()

    # --------------------------------------------------------------- matrizes
    def _lap(self, open_, axis_n):
        """Laplaciano (h² vezes) nas faces abertas de uma componente: vizinhos na direção
        normal (axis_n) são faces abertas (incógnitas) ou de contorno (valor conhecido);
        na direção tangencial, vizinho fechado = parede (célula fantasma, u_g = -u_P)."""
        idx = -np.ones(open_.shape, int)
        idx[open_] = np.arange(open_.sum())
        P = np.argwhere(open_)
        rows, cols, vals = [], [], []
        diag = np.zeros(len(P))
        known = []                                         # (linha, i, j) com valor conhecido
        for ax in (0, 1):
            for s in (-1, 1):
                Q = P.copy(); Q[:, ax] += s
                ok = (Q[:, ax] >= 0) & (Q[:, ax] < open_.shape[ax])
                k = np.full(len(P), -1); k[ok] = idx[Q[ok, 0], Q[ok, 1]]
                inner = k >= 0
                rows.append(np.nonzero(inner)[0]); cols.append(k[inner]); vals.append(np.ones(inner.sum()))
                diag -= 1.0
                if ax == axis_n:                           # face de contorno: valor prescrito
                    kn = np.nonzero(~inner & ok)[0]
                    known += [(r, Q[r, 0], Q[r, 1]) for r in kn]
                else:                                      # parede tangencial: -u_P
                    diag[~inner] -= 1.0
        n = len(P)
        A = sp.csr_matrix((np.concatenate(vals + [diag]),
                           (np.concatenate(rows + [np.arange(n)]), np.concatenate(cols + [np.arange(n)]))),
                          shape=(n, n)) / self.g.h ** 2
        kr = np.array([r for r, _, _ in known], int)
        ki = (np.array([i for _, i, _ in known], int), np.array([j for _, _, j in known], int))
        return A, P, kr, ki

    def _build(self):
        g, dt, nu = self.g, self.dt, self.nu
        self.Au, self.Pu, self.kru, self.kiu = self._lap(self.uo, 0)
        self.Av, self.Pv, self.krv, self.kiv = self._lap(self.vo, 1)
        I = sp.identity
        self.lu_u = spla.splu((I(self.Au.shape[0]) - 0.5 * dt * nu * self.Au).tocsc())
        self.lu_v = spla.splu((I(self.Av.shape[0]) - 0.5 * dt * nu * self.Av).tocsc())
        # Poisson nas células molhadas (Neumann: velocidade prescrita em todo o contorno)
        w = g.wet
        idx = -np.ones(w.shape, int); idx[w] = np.arange(w.sum())
        self.cidx, cells = idx, np.argwhere(w)
        rows, cols, vals, diag = [], [], [], np.zeros(len(cells))
        for di, dj, op in ((1, 0, self.uo[1:]), (-1, 0, self.uo[:-1]), (0, 1, self.vo[:, 1:]),
                           (0, -1, self.vo[:, :-1])):
            o = op[cells[:, 0], cells[:, 1]]
            r = np.nonzero(o)[0]
            rows.append(r); cols.append(idx[cells[r, 0] + di, cells[r, 1] + dj]); vals.append(np.ones(len(r)))
            diag[o] -= 1.0
        n = len(cells)
        M = sp.csr_matrix((np.concatenate(vals + [diag]),
                           (np.concatenate(rows + [np.arange(n)]), np.concatenate(cols + [np.arange(n)]))),
                          shape=(n, n)).tolil()
        i0, j0 = g.celula(g.saida)                         # fixa phi numa célula da barragem
        self.pin = idx[i0, j0]
        M.rows[self.pin] = [self.pin]; M.data[self.pin] = [1.0]
        self.lu_p = spla.splu(M.tocsc())
        self.cells = cells

    # -------------------------------------------------------------- operadores
    def contorno(self, t):
        r = rampa(t)
        self.u[self.ub] = r * self.ubn[self.ub]
        self.v[self.vb] = r * self.vbn[self.vb]

    def bc_vec(self, comp):
        """Contribuição (h² A) dos valores de contorno conhecidos ao laplaciano."""
        if comp == 'u':
            b = np.zeros(self.Au.shape[0]); np.add.at(b, self.kru, self.u[self.kiu])
        else:
            b = np.zeros(self.Av.shape[0]); np.add.at(b, self.krv, self.v[self.kiv])
        return b / self.g.h ** 2

    def convection(self):
        """-(div(u u)) nas faces u e -(div(u v)) nas faces v, central (u = 0 fora do lago)."""
        h, u, v = self.g.h, self.u, self.v
        up = np.pad(u, 1); vp = np.pad(v, 1)
        # x-momento na face u[i, j] (up[i+1, j+1])
        ue = 0.5 * (up[1:-1, 1:-1] + up[2:, 1:-1]); uw = 0.5 * (up[1:-1, 1:-1] + up[:-2, 1:-1])
        un = 0.5 * (up[1:-1, 1:-1] + up[1:-1, 2:]); us = 0.5 * (up[1:-1, 1:-1] + up[1:-1, :-2])
        vn = 0.5 * (vp[:-1, 2:-1] + vp[1:, 2:-1]); vs = 0.5 * (vp[:-1, 1:-2] + vp[1:, 1:-2])
        Cu = -((ue ** 2 - uw ** 2) + (un * vn - us * vs)) / h
        # y-momento na face v[i, j]
        vN = 0.5 * (vp[1:-1, 1:-1] + vp[1:-1, 2:]); vS = 0.5 * (vp[1:-1, 1:-1] + vp[1:-1, :-2])
        vE = 0.5 * (vp[1:-1, 1:-1] + vp[2:, 1:-1]); vW = 0.5 * (vp[1:-1, 1:-1] + vp[:-2, 1:-1])
        uE = 0.5 * (up[2:-1, :-1] + up[2:-1, 1:]); uW = 0.5 * (up[1:-2, :-1] + up[1:-2, 1:])
        Cv = -((vN ** 2 - vS ** 2) + (uE * vE - uW * vW)) / h
        return Cu[self.uo], Cv[self.vo]

    def divergence(self):
        return ((self.u[1:] - self.u[:-1]) + (self.v[:, 1:] - self.v[:, :-1])) / self.g.h

    # ------------------------------------------------------------- escalar
    def _face_tvd(self, C, F, axis):
        """Valor de C nas faces internas ao longo de axis (van Leer); upwind na falta do
        vizinho a montante-montante."""
        Cm = np.moveaxis(C, axis, 0); Fm = np.moveaxis(F, axis, 0)[1:-1]
        wm = np.moveaxis(self.g.wet, axis, 0)
        L, R = Cm[:-1], Cm[1:]
        LL = np.vstack([np.zeros_like(Cm[:1]), Cm[:-2]]); RR = np.vstack([Cm[2:], np.zeros_like(Cm[:1])])
        wLL = np.vstack([np.zeros_like(wm[:1]), wm[:-2]]); wRR = np.vstack([wm[2:], np.zeros_like(wm[:1])])

        def lim(Cu, Cuu, Cd, ok):
            d = Cd - Cu
            r = np.where(np.abs(d) > 1e-30, (Cu - Cuu) / np.where(np.abs(d) > 1e-30, d, 1.0), 0.0)
            psi = np.where(ok, (r + np.abs(r)) / (1 + np.abs(r)), 0.0)
            return Cu + 0.5 * psi * d
        Cf = np.where(Fm >= 0, lim(L, LL, R, wLL), lim(R, RR, L, wRR))
        return np.moveaxis(Cf, 0, axis)

    def escalar(self, dt):
        g, h, k, C = self.g, self.g.h, self.kappa, self.C
        fx = np.zeros_like(self.u); fy = np.zeros_like(self.v)     # fluxo (m²/s · C) em +x, +y
        cx = self._face_tvd(C, self.u, 0)
        fx[1:-1] = (self.u[1:-1] * cx - k * (C[1:] - C[:-1]) / h) * h
        cy = self._face_tvd(C, self.v, 1)
        fy[:, 1:-1] = (self.v[:, 1:-1] * cy - k * (C[:, 1:] - C[:, :-1]) / h) * h
        fx[~self.uo] = 0.0; fy[~self.vo] = 0.0
        # contorno: entradas (Dirichlet: advecção + difusão pela meia célula) e saída (advecção)
        Cl = np.zeros_like(self.u); Cl[1:] = C       # célula à esquerda/direita da face u
        Cr = np.zeros_like(self.u); Cr[:-1] = C
        Cs = np.zeros_like(self.v); Cs[:, 1:] = C
        Cn = np.zeros_like(self.v); Cn[:, :-1] = C
        ent_adv = ent_dif = sai = 0.0
        for f, vel, Cin, CL, CR, out in ((fx, self.u, self.Cin_u, Cl, Cr, self.u_out),
                                         (fy, self.v, self.Cin_v, Cs, Cn, self.v_out)):
            CP = np.where(out > 0, CL, CR)                          # célula molhada
            ent = ~np.isnan(Cin)
            adv = np.where(ent, vel * np.nan_to_num(Cin) * h, 0.0)
            dif = np.where(ent, -k * (np.nan_to_num(Cin) - CP) / (0.5 * h) * h * out, 0.0)
            saida = (out != 0) & ~ent & (vel * out > 0)
            f[ent] = (adv + dif)[ent]
            f[saida] = (vel * CP * h)[saida]
            ent_adv -= np.sum((adv * out)[ent]); ent_dif -= np.sum((dif * out)[ent])
            sai += np.sum((vel * CP * h * out)[saida])
        dC = -((fx[1:] - fx[:-1]) + (fy[:, 1:] - fy[:, :-1])) / h ** 2
        self.C = np.where(g.wet, C + dt * dC, 0.0)
        b = self.balanco
        b['entrada_adv'] += dt * ent_adv; b['entrada_dif'] += dt * ent_dif; b['saida'] += dt * sai

    # ------------------------------------------------------------------ passo
    def iterate(self):
        g, dt, nu, h = self.g, self.dt, self.nu, self.g.h
        Eu, Ev = self.convection()
        if self.prev is None:
            self.prev = (Eu, Ev)
        Pu, Pv = self.prev
        self.prev = (Eu, Ev)
        Eu2, Ev2 = 1.5 * Eu - 0.5 * Pu, 1.5 * Ev - 0.5 * Pv
        uo, vo = self.u[self.uo], self.v[self.vo]
        Lu0 = self.Au @ uo + self.bc_vec('u'); Lv0 = self.Av @ vo + self.bc_vec('v')
        self.contorno(self.t + dt)
        bu1, bv1 = self.bc_vec('u'), self.bc_vec('v')
        gpx = ((self.p[1:] - self.p[:-1]) / h); gpy = ((self.p[:, 1:] - self.p[:, :-1]) / h)
        ru = uo + dt * (Eu2 - gpx_face(gpx, self.uo)) + 0.5 * dt * nu * (Lu0 + bu1)
        rv = vo + dt * (Ev2 - gpy_face(gpy, self.vo)) + 0.5 * dt * nu * (Lv0 + bv1)
        self.u[self.uo] = self.lu_u.solve(ru)
        self.v[self.vo] = self.lu_v.solve(rv)
        # projeção
        rhs = (self.divergence() / dt)[g.wet] * h ** 2
        rhs[self.pin] = 0.0
        phi = np.zeros_like(self.p)
        phi[g.wet] = self.lu_p.solve(rhs)
        self.u[1:-1][self.uo[1:-1]] -= dt * ((phi[1:] - phi[:-1]) / h)[self.uo[1:-1]]
        self.v[:, 1:-1][self.vo[:, 1:-1]] -= dt * ((phi[:, 1:] - phi[:, :-1]) / h)[self.vo[:, 1:-1]]
        self.p += phi
        # escalar com o campo novo, em subpassos pelo CFL e pelo número de difusão
        umax = max(np.abs(self.u).max(), np.abs(self.v).max(), 1e-12)
        n = max(1, int(np.ceil(dt / min(0.4 * h / umax, 0.2 * h * h / max(self.kappa, 1e-30)))))
        for _ in range(n):
            self.escalar(dt / n)
        self.nsub = n
        self.t += dt
        self.iter += 1
        for d in self.dias:
            if d not in self.snaps and self.t >= d * DIA - 1e-6:
                self.snaps[d] = self.C.copy()
        self.serie.append((self.t, self.massa(), *[self.C[i, j] for i, j in self.sondas.values()]))
        div = np.abs(self.divergence()[g.wet]).max() * dt
        self.history.append((self.iter, self.C[self.sondas['Braço do Riacho Fundo']], self.massa() / self.area(), div))

    def massa(self):
        return float(self.C[self.g.wet].sum() * self.g.h ** 2)

    def area(self):
        return float(self.g.wet.sum() * self.g.h ** 2)

    def vazao_saida(self):
        a = self.aberturas['barragem']
        q = sum(self.u[i, j] * self.u_out[i, j] for i, j in a['fu']) + \
            sum(self.v[i, j] * self.v_out[i, j] for i, j in a['fv'])
        return q * self.g.h * H

    def run(self, n_steps, dias=(), report=24, plot_file=None):
        self.dias, self.serie = tuple(dias), []
        mon = ConvergenceMonitor(['C sonda RF', 'C média', 'max|div|dt'], report=report, logy=False,
                                 ylabel='valor', dt=self.dt / DIA, plot_file=plot_file,
                                 title='Lago Paranoá — FVM (t em dias)')
        return run_loop(self, n_steps, mon)


def gpx_face(gpx, uo):
    return gpx[uo[1:-1]]


def gpy_face(gpy, vo):
    return gpy[vo[:, 1:-1]]


# ------------------------------------------------------------------- figuras
def _contorno(ax, g):
    a = np.vstack([g.anel, g.anel[:1]]) / 1e3
    ax.plot(a[:, 0], a[:, 1], color='#0b0b0b', lw=0.7)
    ax.set_aspect('equal'); ax.set_xlabel('x UTM 23S (km)'); ax.set_ylabel('y UTM 23S (km)')


def plot_dominio(s, fname):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g = s.g
    ext = np.array([g.x0, g.x0 + g.NX * g.h, g.y0, g.y0 + g.NY * g.h]) / 1e3
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(np.where(g.wet, 1.0, np.nan).T, origin='lower', extent=ext, cmap='Blues', vmin=0, vmax=4)
    _contorno(ax, g)
    for nome, a in s.aberturas.items():
        pts = [(g.x0 + i * g.h, g.yc[j]) for i, j in a['fu']] + [(g.xc[i], g.y0 + j * g.h) for i, j in a['fv']]
        pts = np.array(pts) / 1e3
        c = '#eb6834' if nome == 'barragem' else ('#e87ba4' if nome == FONTE else '#2a78d6')
        ax.plot(pts[:, 0], pts[:, 1], 's', color=c, ms=5)
        rot = (f"barragem: Q = {a['Q']:g} m³/s (saída)" if nome == 'barragem' else
               f"{nome.replace('_', ' ')}: Q = {a['Q']:g} m³/s, C = {a['C']:g}")
        ax.annotate(rot, pts[0], textcoords='offset points', xytext=(8, 6), fontsize=8)
    for nome, (i, j) in s.sondas.items():
        ax.plot(g.xc[i] / 1e3, g.yc[j] / 1e3, 'D', color='#1baf7a', mec='white', ms=7)
        ax.annotate(nome, (g.xc[i] / 1e3, g.yc[j] / 1e3), textcoords='offset points', xytext=(6, -10), fontsize=8)
    ax.text(0.02, 0.62, f"margens: u = v = 0, ∂C/∂n = 0\nH = {H:g} m, ν = {s.nu:g} m²/s, κ = {s.kappa:g} m²/s\n"
            f"vazões: rampa de 0 ao nominal em 1 dia\ncélulas fluidas: centro dentro do contorno",
            transform=ax.transAxes, fontsize=8, va='top')
    ax.set_title('Lago Paranoá — domínio e condições de contorno (FVM)\n'
                 '(contorno: © OpenStreetMap contributors, ODbL)')
    fig.savefig(fname, dpi=130, bbox_inches='tight'); plt.close(fig)


def plot_malha(s, fname):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g = s.g
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    for ax, zoom in zip(axs, (None, g.entradas[FONTE])):
        if zoom is None:
            i0, i1, j0, j1 = 0, g.NX, 0, g.NY
        else:
            ic, jc = g.celula(zoom)
            i0, i1, j0, j1 = max(ic - 12, 0), min(ic + 13, g.NX), max(jc - 9, 0), min(jc + 10, g.NY)
        xs = (g.x0 + np.arange(i0, i1 + 1) * g.h) / 1e3; ys = (g.y0 + np.arange(j0, j1 + 1) * g.h) / 1e3
        ax.pcolormesh(xs, ys, np.where(g.wet[i0:i1, j0:j1], 1.0, np.nan).T, cmap='Blues', vmin=0, vmax=3,
                      edgecolors='#5f5e5a' if zoom is not None else 'none', lw=0.3)
        _contorno(ax, g)
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
        if zoom is not None:
            a = s.aberturas[FONTE]
            for i, j in a['fu']:
                ax.plot([(g.x0 + i * g.h) / 1e3] * 2, [(g.y0 + j * g.h) / 1e3, (g.y0 + (j + 1) * g.h) / 1e3], color='#e87ba4', lw=3)
            for i, j in a['fv']:
                ax.plot([(g.x0 + i * g.h) / 1e3, (g.x0 + (i + 1) * g.h) / 1e3], [(g.y0 + j * g.h) / 1e3] * 2, color='#e87ba4', lw=3)
            ax.set_title('Foz do Riacho Fundo: faces de entrada (rosa)')
        else:
            ax.set_title(f'Malha: {g.NX} × {g.NY} células de {g.h:g} m, {g.wet.sum()} no lago')
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def plot_resultados(s, fname):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    g = s.g
    azul = LinearSegmentedColormap.from_list('azul', ['#f4f8fd', '#cde2fb', '#9ec5f4', '#6da7ec',
                                                      '#3987e5', '#256abf', '#184f95', '#0d366b'])
    ext = np.array([g.x0, g.x0 + g.NX * g.h, g.y0, g.y0 + g.NY * g.h]) / 1e3
    snaps = dict(sorted(s.snaps.items())) or {round(s.t / DIA, 2): s.C}
    fig, axs = plt.subplots(2, len(snaps), figsize=(4.2 * len(snaps), 9), squeeze=False)
    ic, jc = g.celula(g.entradas[FONTE])
    zx = g.xc[ic] / 1e3 + np.array([-0.6, 3.4]); zy = g.yc[jc] / 1e3 + np.array([-1.2, 2.0])
    for k, (d, C) in enumerate(snaps.items()):
        for row, (xl, yl) in enumerate(((None, None), (zx, zy))):
            ax = axs[row, k]
            im = ax.imshow(np.ma.masked_where(~g.wet, np.clip(C, 1e-5, 1)).T, origin='lower', extent=ext,
                           norm=LogNorm(1e-4, 1), cmap=azul)
            _contorno(ax, g)
            if xl is not None:
                ax.set_xlim(*xl); ax.set_ylim(*yl)
            ax.set_title(f'{d:g} dias' + (' (foz do Riacho Fundo)' if xl is not None else ''), fontsize=10)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label='C/C0 (escala log)')
    fig.suptitle('FVM: poluente lançado pelo Riacho Fundo (C/C0 = 1 na foz)')
    fig.savefig(fname, dpi=130, bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='Lago Paranoá: caso paranoa do Cordel em volumes finitos')
    ap.add_argument('--dias', type=float, default=7.0)
    ap.add_argument('--h', type=float, default=80.0, help='tamanho da célula (m)')
    ap.add_argument('--dt', type=float, default=1800.0, help='passo de tempo do escoamento (s)')
    ap.add_argument('--nu', type=float, default=NU, help='viscosidade horizontal (m²/s)')
    ap.add_argument('--kappa', type=float, default=KAPPA, help='difusividade do poluente (m²/s)')
    ap.add_argument('--out', default=str(REPO / 'results'))
    ap.add_argument('--apenas-malha', action='store_true')
    a = ap.parse_args()
    nome = f'paranoa_{a.dias:g}d_h{a.h:g}' + (f'_k{a.kappa:g}' if a.kappa != KAPPA else '') \
        + (f'_nu{a.nu:g}' if a.nu != NU else '')
    cd = CaseDir(a.out, nome)
    cd.announce()
    g = LagoGrid(a.h)
    t0 = time.perf_counter()
    s = ParanoaFVM(g, dt=a.dt, nu=a.nu, kappa=a.kappa)
    t_montagem = time.perf_counter() - t0
    print(f'  malha {g.NX} x {g.NY}, {g.wet.sum()} células no lago ({s.area() / 1e6:.2f} km²), '
          f'{s.uo.sum() + s.vo.sum()} faces de velocidade, montagem {t_montagem:.1f} s')
    for nome_a, ab in s.aberturas.items():
        print(f"     {nome_a:16s} Q = {ab['Q']:4.1f} m³/s, {len(ab['fu']) + len(ab['fv'])} faces, "
              f"U = {ab['U'] * 100:.3f} cm/s (foz a {ab['dist']:.0f} m da face)")
    plot_dominio(s, cd.dominio); plot_malha(s, cd.malha)
    if a.apenas_malha:
        return
    n = int(round(a.dias * DIA / a.dt))
    dias = [d for d in (1, 2, 4, 7, 14, 30, 90, 180, 365) if d <= a.dias + 1e-9]
    s.run(n, dias=dias, plot_file=cd.convergencia)
    S = np.array(s.serie)
    b = s.balanco
    bal = (s.massa() + b['saida'] - b['entrada_adv'] - b['entrada_dif']) / max(b['entrada_adv'] + b['entrada_dif'], 1e-30)
    bal_adv = (s.massa() + b['saida'] - b['entrada_adv']) / max(b['entrada_adv'], 1e-30)
    print(f'  t = {s.t / DIA:.2f} dias, {s.iter} passos de {a.dt:g} s (subpassos do escalar: {s.nsub}), '
          f'{s.seconds:.1f} s')
    print(f'  vazão na barragem = {s.vazao_saida():.3f} m³/s (soma das entradas {sum(VAZOES.values()):.2f})')
    print(f"  poluente: no lago {s.massa() / s.area():.4f} (C/C0 média); entrou por advecção "
          f"{b['entrada_adv']:.4g} m², por difusão {b['entrada_dif']:.4g} m², saiu {b['saida']:.4g} m²")
    print(f'  (lago + saída - entrada total) / entrada = {bal:+.2e};  (lago + saída - entrada advectiva) / '
          f'entrada advectiva = {bal_adv:+.2%}')
    for (k, _), c in zip(s.sondas.items(), S[-1, 2:]):
        print(f'   C/C0 em {k:22s} = {c:.4f}')
    print(f'  C mín/máx no lago: {s.C[g.wet].min():.2e} / {s.C[g.wet].max():.4f}')
    plot_resultados(s, cd.resultado())
    uc = 0.5 * (s.u[1:] + s.u[:-1]); vc = 0.5 * (s.v[:, 1:] + s.v[:, :-1])
    np.savez(cd.campos, x=g.xc, y=g.yc, wet=g.wet, C=s.C, u=uc, v=vc, p=s.p, serie=S,
             nomes_sondas=list(s.sondas), dt=a.dt, h=a.h, seconds=s.seconds,
             **{f'C_{d:g}d': c for d, c in s.snaps.items()}, **{f'bal_{k}': v for k, v in b.items()})
    cd.save_json(dict(lago='Paranoá', h=a.h, dt=a.dt, dias=a.dias, NX=int(g.NX), NY=int(g.NY),
                      cells=int(g.wet.sum()), faces_vel=int(s.uo.sum() + s.vo.sum()), iter=s.iter,
                      nsub=s.nsub, seconds=round(s.seconds, 1), seconds_montagem=round(t_montagem, 2),
                      Q_saida=s.vazao_saida(), C_media=s.massa() / s.area(),
                      entrada_adv=b['entrada_adv'], entrada_dif=b['entrada_dif'], saida=b['saida'],
                      balanco_rel=bal, balanco_adv_rel=bal_adv, C_min=float(s.C[g.wet].min()),
                      C_max=float(s.C[g.wet].max()),
                      sondas={k: float(c) for k, c in zip(s.sondas, S[-1, 2:])}))


if __name__ == '__main__':
    main()
