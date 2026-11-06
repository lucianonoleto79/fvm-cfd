"""
Cilindro 2D, Re = 100, em DOMÍNIO RETANGULAR [-Lu, Ld] x [-H, H] (em D) — malha cartesiana
esticada + fronteira imersa (cartesian.py). Serve para estudar o efeito do tamanho do
domínio e comparar com a malha O (cylinder.py).

Domínios predefinidos (a malha perto do cilindro é idêntica em todos):
    P  [-10, 20] x [-10, 10]     bloqueio D/2H = 5 %
    M  [-20, 40] x [-20, 20]     2,5 %
    G  [-40, 80] x [-40, 40]     1,25 %

Contornos: entrada (x = -Lu) u = U, v = 0; laterais (y = ±H) deslizamento livre;
saída (x = Ld) convectiva. Mesma rotação inicial breve da malha O para disparar a emissão.

Uso:
    python cylinder_rect.py --domain M              # um domínio
    python cylinder_rect.py --domain P M G          # estudo do tamanho do domínio
    python cylinder_rect.py --domain M --mesh-only  # só a figura da malha
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cartesian import CartesianGrid, CartesianIBNS, Cylinder
from control import (add_control_args, clear_stale_commands, control_from_args,
                     maybe_resume, save_checkpoint)
from cylinder import LITERATURE, pulse, stats
from monitor import ConvergenceMonitor, add_monitor_args

DEFAULT_OUT = Path(__file__).resolve().parent.parent / 'results'
DOMAINS = {'P': (10, 20, 10), 'M': (20, 40, 20), 'G': (40, 80, 40)}


def plot_mesh(g, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    col = dict(colors='#2b5d8a', linewidths=0.3)

    def lines(ax, every=1, box=None):
        xf, yf = g.xf, g.yf
        if box:
            xf = xf[(xf >= box[0]) & (xf <= box[1])]; yf = yf[(yf >= box[2]) & (yf <= box[3])]
        x0, x1, y0, y1 = (xf[0], xf[-1], yf[0], yf[-1])
        segs = [[(x, y0), (x, y1)] for x in xf[::every]] + [[(x0, y), (x1, y)] for y in yf[::every]]
        ax.add_collection(LineCollection(segs, **col))
        return x0, x1, y0, y1

    x0, x1, y0, y1 = lines(axs[0], every=4)
    axs[0].set_xlim(x0 - 1, x1 + 1); axs[0].set_ylim(y0 - 1, y1 + 1)
    axs[0].set_title(f'(a) domínio [{g.xf[0]:g}, {g.xf[-1]:g}] × [{g.yf[0]:g}, {g.yf[-1]:g}] D\n'
                     f'(1 de cada 4 linhas)', fontsize=10)
    lines(axs[1], every=2, box=(-3, 14, -4, 4))
    axs[1].set_xlim(-3, 14); axs[1].set_ylim(-4, 4)
    axs[1].set_title('(b) núcleo uniforme e refino da esteira\n(1 de cada 2 linhas)', fontsize=10)
    lines(axs[2], box=(-0.8, 0.0, -0.4, 0.4))
    axs[2].set_xlim(-0.75, -0.25); axs[2].set_ylim(-0.25, 0.25)
    axs[2].set_title(f'(c) fronteira imersa junto à parede\nh = {g.dx.min():.4f} D (o círculo corta as células)',
                     fontsize=10)
    for ax in axs:
        ax.add_patch(Circle((0, 0), 0.5, fill=ax is not axs[2], color='0.25', zorder=3,
                            lw=1.5 if ax is axs[2] else 0))
        ax.set_aspect('equal'); ax.set_xlabel('x / D'); ax.set_ylabel('y / D')
    fig.suptitle(f'Malha cartesiana esticada: {g.NX} × {g.NY} = {g.cells} volumes', fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_case(s, res, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    g = s.g
    h = np.array(s.history)
    t = h[:, 0] * s.dt
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15])
    a = fig.add_subplot(gs[0])
    a.plot(t, h[:, 1], lw=0.9, label='$C_d$'); a.plot(t, h[:, 2], lw=0.9, label='$C_l$')
    if res:
        a.axvspan(res['t_from'], t[-1], color='0.92', zorder=0, label='janela das médias')
        st = f"{res['St']:.4f}" if res['St'] else '-'
        a.set_title(f"$\\overline{{C_d}}$ = {res['Cd_mean']:.4f} ± {res['Cd_amp']:.4f},  "
                    f"$C_l$ = ± {res['Cl_amp']:.4f},  St = {st}", fontsize=10)
    a.set_ylim(-0.8, 2.2); a.set_xlabel('t U / D'); a.grid(alpha=0.3); a.legend(fontsize=8, ncol=3)
    c = fig.add_subplot(gs[1])
    w = s.vorticity()
    X, Y = np.meshgrid(g.xf[1:-1], g.yf[1:-1], indexing='ij')
    sel = (g.xf[1:-1] > -3) & (g.xf[1:-1] < 21)
    selY = (g.yf[1:-1] > -5) & (g.yf[1:-1] < 5)
    W = np.clip(w[np.ix_(sel, selY)], -3, 3)
    XX, YY = X[np.ix_(sel, selY)], Y[np.ix_(sel, selY)]
    W[np.hypot(XX, YY) < 0.5] = np.nan
    cs = c.contourf(XX, YY, W, levels=np.linspace(-3, 3, 25), cmap='RdBu_r', extend='both')
    c.add_patch(Circle((0, 0), 0.5, color='0.3', zorder=3))
    c.set_xlim(-2, 20); c.set_ylim(-4, 4); c.set_aspect('equal')
    c.set_xlabel('x / D'); c.set_ylabel('y / D')
    c.set_title(f'Vorticidade $\\omega_z D/U$ em t = {s.t:.1f}', fontsize=10)
    fig.colorbar(cs, ax=c, shrink=0.8, pad=0.01)
    fig.suptitle(f'Cilindro, Re = {s.Re:g}, domínio retangular [{g.xf[0]:g}, {g.xf[-1]:g}] × '
                 f'[{g.yf[0]:g}, {g.yf[-1]:g}] D, {g.NX} × {g.NY} volumes', fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Cilindro 2D em domínio retangular (fronteira imersa)')
    ap.add_argument('--Re', type=float, default=100)
    ap.add_argument('--domain', nargs='+', default=['M'],
                    help='P, M, G ou Lu,Ld,H (ex.: 15,30,15)')
    ap.add_argument('--h', type=float, default=0.025, help='espaçamento no núcleo, em D')
    ap.add_argument('--T', type=float, default=150)
    ap.add_argument('--cfl', type=float, default=0.35)
    ap.add_argument('--stats-from', type=float, default=None)
    ap.add_argument('--mesh-only', action='store_true')
    add_monitor_args(ap, report=2000)
    add_control_args(ap)
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    clear_stale_commands(args, OUT)

    results = []
    for dom in args.domain:
        Lu, Ld, H = DOMAINS[dom] if dom in DOMAINS else map(float, dom.split(','))
        g = CartesianGrid.for_cylinder(Lu, Ld, H, h=args.h)
        tag = f'{dom}' if dom in DOMAINS else f'{Lu:g}_{Ld:g}_{H:g}'
        case = f'cilindro_ret_{tag}_h{args.h:g}'
        print(f'\n=== Cilindro Re = {args.Re:g}, domínio {tag}: {g.describe()} ===', flush=True)
        plot_mesh(g, OUT / f'malha_ret_{tag}.png')
        if args.mesh_only:
            continue
        s = CartesianIBNS(g, Cylinder(0.5, pulse), Re=args.Re, cfl=args.cfl)
        n_steps = int(round(args.T / s.dt))
        print(f'  dt = {s.dt:.5f}, {n_steps} passos; nós fantasmas u/v = {len(s.u_rows)}/{len(s.v_rows)}',
              flush=True)
        ckpt = OUT / f'{case}_parcial.npz'
        maybe_resume(args, s, ckpt)
        t_from = args.stats_from if args.stats_from is not None else args.T / 2
        mon = ConvergenceMonitor(['Cd', 'Cl'], report=args.report,
                                 plot_file=None if args.no_curve else OUT / f'forcas_{case}.png',
                                 live=args.live, refresh=args.refresh, logy=False, ylabel='coeficiente',
                                 dt=s.dt, title=f'Cilindro retangular {tag}')

        def partial(s, case=case):
            plot_case(s, stats(s, min(t_from, s.t / 2)) if s.iter > 10 else None, OUT / f'{case}_parcial.png')

        s.run(n_steps, monitor=mon, control=control_from_args(args, OUT, ckpt, on_save=partial))
        res = stats(s, min(t_from, s.t / 2))
        res.update(malha='retangular', domain=tag, Lu=Lu, Ld=Ld, H=H, blockage=0.5 / H, h=args.h,
                   NX=g.NX, NY=g.NY, cells=g.cells, dt=s.dt, t=s.t, steps=s.iter, cfl=s.cfl(),
                   seconds=round(getattr(s, 'seconds_total', 0.0), 1),
                   interrompido=bool(getattr(s, 'stopped', False)))
        results.append(res)
        plot_case(s, res, OUT / f'{case}.png')
        save_checkpoint(s, OUT / f'{case}.npz')
        if getattr(s, 'stopped', False):
            print('\n  simulação interrompida: continue com o mesmo comando + --resume', flush=True)
            break

    if not results:
        return
    print('\n  domínio            volumes  bloqueio  ciclos  Cd médio  amp Cd   amp Cl    St      tempo')
    for r in results:
        st = f"{r['St']:.4f}" if r['St'] else '  -   '
        print(f"  {r['domain']:16s} {r['cells']:8d}   {100 * r['blockage']:5.2f} %   {r['ciclos']:3d}   "
              f"{r['Cd_mean']:.4f}   {r['Cd_amp']:.4f}  {r['Cl_amp']:.4f}  {st}  {r['seconds']:.0f} s")
    print('\n  Referências (confira nas fontes):')
    for name, cd, cda, cla, st in LITERATURE:
        f = lambda v: f'{v:.3f}' if v is not None else '  -  '
        print(f'    {name:42s} Cd {f(cd)}  amp Cd {f(cda)}  amp Cl {f(cla)}  St {f(st)}')
    with open(OUT / f'cilindro_ret_Re{args.Re:g}.json', 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n  abortado pelo usuário (2º Ctrl+C); o último checkpoint *_parcial.npz foi mantido',
              flush=True)
        raise SystemExit(130)
