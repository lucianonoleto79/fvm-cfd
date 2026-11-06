"""
pipe.py — Validação do solver 3D (fvm3d.py): escoamento laminar em tubo circular
-------------------------------------------------------------------------------
Tubo de diâmetro D alinhado com z, perfil uniforme U na entrada (z = 0) e saída
livre (z = L). A parede circular é uma fronteira imersa na malha cartesiana.

Referências analíticas / da literatura:
  * Hagen-Poiseuille (plenamente desenvolvido):
        w(r) = 2 U (1 - (2r/D)^2),   f = -(dp/dz) D / (rho U^2 / 2) = 64 / Re
  * Comprimento de entrada hidrodinâmico (centro atinge 99% de 2U):
        Durst, Ray, Ünsal & Bayoumi, J. Fluids Eng. 127 (2005) 1154-1160:
        L/D = [0.619^1.6 + (0.0567 Re)^1.6]^(1/1.6)

Uso:
    python pipe.py                  # estudo de malha: 11, 15, 21 células no diâmetro
    python pipe.py --n 15 --Re 100
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fvm3d import Grid3D, SIMPLEC3D

DEFAULT_OUT = Path(__file__).resolve().parent.parent / 'results'

D, R, U, RHO = 1.0, 0.5, 1.0, 1.0


def durst_entrance_length(Re):
    return (0.619 ** 1.6 + (0.0567 * Re) ** 1.6) ** (1 / 1.6)


def build(n, Re, L, nz_per_D, **kw):
    """n (ímpar) células no diâmetro -> eixo do tubo passa por centros de células."""
    if n % 2 == 0:
        raise ValueError('use n ímpar (eixo do tubo sobre um centro de célula)')
    h = D / n
    m = n + 2                                     # uma célula de folga de cada lado
    half = m * h / 2
    nz = int(round(L * nz_per_D))
    g = Grid3D((m, m, nz), (-half, -half, 0.0), (half, half, L * D))
    phi = lambda X, Y, Z: np.sqrt(X ** 2 + Y ** 2) - R
    fluid2d = phi(*np.meshgrid(g.centers(0), g.centers(1), indexing='ij'), 0) < 0
    # a continuidade enxerga a seção em escada; escala U_in para que a vazão seja U*pi*R^2
    U_in = U * np.pi * R ** 2 / (fluid2d.sum() * h * h)
    bc = {(2, -1): ('dirichlet', (0.0, 0.0, U_in)), (2, 1): ('outlet',)}
    s = SIMPLEC3D(g, mu=RHO * U * D / Re, rho=RHO, phi=phi, bc=bc, Uref=U, **kw)
    s.V[2][...] = np.where(s.fluid[:, :, :1], U_in, 0.0) * np.ones(s.V[2].shape)  # chute inicial
    return s


def analyze(s, Re, L):
    g = s.g
    w = 0.5 * (s.V[2][:, :, :-1] + s.V[2][:, :, 1:])        # w nos centros das células
    X, Y, Z = g.coords()
    r = np.sqrt(X ** 2 + Y ** 2)
    zc = g.centers(2)
    ic = g.n[0] // 2                                         # célula sobre o eixo
    wc = w[ic, ic, :]                                        # velocidade na linha de centro
    # perfil plenamente desenvolvido em z = 0.9 L (nós w nas faces z)
    kz = int(round(0.9 * g.n[2]))
    wf = s.V[2][:, :, kz]
    # comprimento de entrada: wc atinge 99% do valor desenvolvido (da própria solução)
    target = 0.99 * wf[ic, ic]
    k = int(np.argmax(wc >= target))
    Le = np.interp(target, wc[k - 1:k + 1], zc[k - 1:k + 1]) if k > 0 else np.nan
    fl = s.fluid[:, :, kz]
    wex = 2 * U * (1 - (2 * r[:, :, 0] / D) ** 2)
    err = (wf - wex)[fl] / (2 * U)
    # gradiente de pressão na região desenvolvida: média da seção, ajuste linear
    pm = np.array([s.p[:, :, k][s.fluid[:, :, k]].mean() for k in range(g.n[2])])
    sel = (zc > 0.7 * L) & (zc < 0.95 * L)
    dpdz = np.polyfit(zc[sel], pm[sel], 1)[0]
    f = -dpdz * D / (0.5 * RHO * U ** 2)
    Q = s.V[2][:, :, kz].sum() * g.area[2]
    return dict(n=g.n[0] - 2, h=float(g.h[0]), cells=int(np.prod(g.n)), fluid_cells=int(s.fluid.sum()),
                iter=s.iter, converged=s.converged, seconds=round(s.seconds, 1),
                wc_max=float(wf[ic, ic]), wc_err_pct=float(100 * (wf[ic, ic] - 2 * U) / (2 * U)),
                prof_rms=float(np.sqrt(np.mean(err ** 2))), prof_max=float(np.abs(err).max()),
                fRe=float(f * Re), fRe_err_pct=float(100 * (f * Re - 64) / 64),
                Le=float(Le), Le_durst=durst_entrance_length(Re),
                Q_ratio=float(Q / (U * np.pi * R ** 2))), dict(zc=zc, wc=wc, pm=pm, sel=sel, dpdz=dpdz, kz=kz)


def plot(s, res, aux, study, Re, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    g = s.g
    blue, red, green = '#2f6fd6', '#d9532b', '#2e8a4e'
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
    ic, kz = g.n[0] // 2, aux['kz']

    # (a) perfil desenvolvido: linhas x e y pelo centro + todos os nós vs r
    a = ax[0, 0]
    X, Y, _ = g.coords(2)
    r = np.sqrt(X[:, :, kz] ** 2 + Y[:, :, kz] ** 2)
    fl = s.fluid[:, :, kz]
    rr = np.linspace(0, R, 100)
    a.plot(2 * r[fl] / D, s.V[2][:, :, kz][fl] / U, '.', color=blue, ms=4, alpha=0.5, label='FVM (todos os nós)')
    a.plot(2 * rr / D, 2 * (1 - (2 * rr / D) ** 2), '-', color=red, lw=2, label='Hagen–Poiseuille')
    a.set(xlabel='2r / D', ylabel='w / U', title=f'Perfil em z = {g.faces(2)[kz]:.1f} D')
    a.legend(frameon=False); a.grid(alpha=0.3)

    # (b) seção transversal
    a = ax[0, 1]
    xc = g.centers(0)
    wf = np.where(fl, s.V[2][:, :, kz], np.nan)
    cs = a.contourf(xc, xc, wf.T, levels=np.linspace(0, 2, 21), cmap='viridis')
    a.add_patch(Circle((0, 0), R, fill=False, color='w', lw=1.5))
    a.set(aspect='equal', title='w / U na seção (círculo = parede real)', xlabel='x / D', ylabel='y / D')
    fig.colorbar(cs, ax=a, shrink=0.8)

    # (c) corte longitudinal y = 0
    a = ax[0, 2]
    wl = 0.5 * (s.V[2][:, ic, :-1] + s.V[2][:, ic, 1:])
    wl = np.where(s.fluid[:, ic, :], wl, np.nan)
    cs = a.contourf(aux['zc'], xc, wl, levels=np.linspace(0, 2, 21), cmap='viridis')
    a.axvline(res['Le'], color='w', ls='--', lw=1)
    a.set(title='w / U no plano y = 0 (entrada uniforme → parabólico)', xlabel='z / D', ylabel='x / D')
    fig.colorbar(cs, ax=a, shrink=0.8)

    # (d) desenvolvimento da velocidade no centro
    a = ax[1, 0]
    a.plot(aux['zc'], aux['wc'] / U, color=blue, lw=2, label='FVM')
    a.axhline(2, color='0.5', lw=0.8)
    a.axhline(1.98, color='0.5', lw=0.8, ls=':')
    a.axvline(res['Le_durst'], color=red, ls='--', label=f"Durst et al. (2005): L/D = {res['Le_durst']:.2f}")
    a.axvline(res['Le'], color=blue, ls=':', label=f"FVM: L/D = {res['Le']:.2f}")
    a.set(xlabel='z / D', ylabel='w_c / U', title=f'Velocidade no eixo (Re = {Re})')
    a.legend(frameon=False, loc='lower right'); a.grid(alpha=0.3)

    # (e) pressão média na seção
    a = ax[1, 1]
    zc, pm, sel = aux['zc'], aux['pm'], aux['sel']
    a.plot(zc, pm - pm[-1], color=blue, lw=2, label='FVM (média na seção)')
    zf = zc[sel]
    p_fd = -64 / Re * 0.5 * RHO * U ** 2 / D
    a.plot(zf, pm[sel][0] - pm[-1] + p_fd * (zf - zf[0]), '--', color=red, lw=2,
           label=f'inclinação 64/Re (f·Re FVM = {res["fRe"]:.2f})')
    a.set(xlabel='z / D', ylabel='p − p(saída)', title='Queda de pressão')
    a.legend(frameon=False); a.grid(alpha=0.3)

    # (f) estudo de malha
    a = ax[1, 2]
    if len(study) > 1:
        hh = np.array([q['h'] for q in study])
        for key, lab, col in (('prof_rms', 'erro RMS do perfil', blue),
                              ('wc_err_pct', '|erro| w_c (%)', green),
                              ('fRe_err_pct', '|erro| f·Re (%)', red)):
            e = np.abs([q[key] for q in study])
            if key == 'prof_rms':
                e = 100 * e
            a.loglog(hh, e, 'o-', color=col, label=lab + (' (%)' if key == 'prof_rms' else ''))
        a.loglog(hh, 100 * study[0]['prof_rms'] * (hh / hh[0]) ** 2, 'k:', label='inclinação 2')
        a.set(xlabel='h / D', ylabel='erro (%)', title='Convergência de malha')
        a.legend(frameon=False, fontsize=9); a.grid(alpha=0.3, which='both')
    else:
        a.axis('off')

    vols = f'{res["cells"]:,}'.replace(',', '.')
    fig.suptitle(f'Tubo circular, Re = {Re:g}: solver 3D FVM/SIMPLEC com fronteira imersa '
                 f'({res["n"]} células no diâmetro, {vols} volumes)')
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Escoamento laminar em tubo circular (validação 3D)')
    ap.add_argument('--n', type=int, nargs='+', default=[11, 15, 21], help='células no diâmetro (ímpar)')
    ap.add_argument('--Re', type=float, default=100)
    # perfil a 99% do desenvolvido em ~6D (Re = 100), mas f e w_c só ficam a 0,1% do
    # valor de Poiseuille depois de ~14D: o tubo precisa ser longo
    ap.add_argument('--L', type=float, default=20, help='comprimento em diâmetros')
    ap.add_argument('--nz-per-D', type=float, default=10)
    ap.add_argument('--tol', type=float, default=1e-6)
    ap.add_argument('--max-iter', type=int, default=3000)
    ap.add_argument('--reanalyze', action='store_true', help='refaz análise/figuras a partir dos .npz')
    ap.add_argument('--out', default=str(DEFAULT_OUT), help='pasta de saída (padrão: results/ na raiz do repositório)')
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    study, last = [], None
    old = {}
    if args.reanalyze:
        with open(OUT / f'tubo_Re{args.Re:g}.json', encoding='utf-8') as f:
            old = {q['n']: q for q in json.load(f)}
    for n in args.n:
        s = build(n, args.Re, args.L, args.nz_per_D, tol=args.tol)
        if args.reanalyze:
            d = np.load(OUT / f'tubo_Re{args.Re:g}_n{n}.npz')
            s.V = [d['u'], d['v'], d['w']]; s.p = d['p']
            s.history = d['history'].tolist(); s.iter = len(s.history)
            s.converged, s.seconds = old[n]['converged'], old[n]['seconds']
        else:
            print(f'\n=== Tubo: Re = {args.Re:g}, {n} células no diâmetro, L = {args.L:g} D ===', flush=True)
            s.run(max_iter=args.max_iter)
        res, aux = analyze(s, args.Re, args.L)
        study.append(res)
        last = (s, res, aux)
        if not args.reanalyze:
            np.savez(OUT / f'tubo_Re{args.Re:g}_n{n}.npz', u=s.V[0], v=s.V[1], w=s.V[2], p=s.p,
                 history=np.array(s.history))
    s, res, aux = last
    plot(s, res, aux, study, args.Re, OUT / f'tubo_Re{args.Re:g}.png')

    print(f"\n  n     h      volumes  iter     w_c/2U-1   perfil RMS  perfil máx   f·Re    erro     "
          f"L/D (FVM)  L/D (Durst)  Q/Q0   tempo")
    for q in study:
        print(f"{q['n']:3d}  {q['h']:.4f}  {q['cells']:8d}  {q['iter']:4d}  {q['wc_err_pct']:+8.3f} %  "
              f"{100 * q['prof_rms']:8.3f} %  {100 * q['prof_max']:8.3f} %  {q['fRe']:6.2f}  "
              f"{q['fRe_err_pct']:+6.2f} %  {q['Le']:8.2f}  {q['Le_durst']:10.2f}  {q['Q_ratio']:.4f}  "
              f"{q['seconds']:.0f} s")
    if len(study) > 1:
        for key in ('prof_rms', 'fRe_err_pct', 'wc_err_pct'):
            e = np.abs([q[key] for q in study]); h = np.array([q['h'] for q in study])
            p = np.polyfit(np.log(h), np.log(e), 1)[0]
            print(f'  ordem observada ({key}): {p:.2f}')
    with open(OUT / f'tubo_Re{args.Re:g}.json', 'w', encoding='utf-8') as f:
        json.dump(study, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
