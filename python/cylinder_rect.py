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
from casefiles import CaseDir, study_dir
from monitor import ConvergenceMonitor, add_monitor_args

DEFAULT_OUT = Path(__file__).resolve().parent.parent / 'results'
DOMAINS = {'P': (10, 20, 10), 'M': (20, 40, 20), 'G': (40, 80, 40)}


def plot_domain(g, fname, box=(-1.25, 1.25, -1.25, 1.25)):
    """Domínio retangular e condições de contorno (em escala)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
    x0, x1, y0, y1 = g.xf[0], g.xf[-1], g.yf[0], g.yf[-1]
    W, H = x1 - x0, y1 - y0
    fig, ax = plt.subplots(figsize=(12, 12 * H / W * 0.75 + 2.2))
    ax.add_patch(Rectangle((x0, y0), W, H, color='#f4f8fc', zorder=0))
    ax.plot([x0, x0], [y0, y1], color='#1f6fb4', lw=5)
    ax.plot([x1, x1], [y0, y1], color='#c0392b', lw=5)
    ax.plot([x0, x1], [y1, y1], color='#7d7d7d', lw=4, ls='--')
    ax.plot([x0, x1], [y0, y0], color='#7d7d7d', lw=4, ls='--')
    for y in np.linspace(y0 + 0.15 * H, y1 - 0.15 * H, 5):
        ax.add_patch(FancyArrowPatch((x0 - 0.09 * W, y), (x0 - 0.01 * W, y), arrowstyle='-|>',
                                     mutation_scale=14, color='#1f6fb4', clip_on=False))
    ax.add_patch(Circle((0, 0), 0.5 * max(1.0, 0.012 * W), color='0.15', zorder=3))
    ax.add_patch(Rectangle((-1.5, -1.5), 3, 3, fill=False, ec='#2e8a4e', lw=1.2, ls=':'))
    kw = dict(fontsize=10, va='center')
    ax.text(x0 + 0.02 * W, y1 - 0.08 * H, 'ENTRADA (x = −Lu): u = U, v = 0  (Dirichlet)', color='#1f6fb4', **kw)
    ax.text(x1 - 0.02 * W, y1 - 0.08 * H, 'SAÍDA (x = Ld): convectiva\n∂u/∂t + U ∂u/∂x = 0, vazão corrigida',
            color='#c0392b', ha='right', **kw)
    ax.text(0.5 * (x0 + x1), y1 + 0.04 * H, 'LATERAL (y = +H): deslizamento livre  v = 0, ∂u/∂y = 0',
            color='0.35', ha='center', **kw)
    ax.text(0.5 * (x0 + x1), y0 - 0.04 * H, 'LATERAL (y = −H): deslizamento livre  v = 0, ∂u/∂y = 0',
            color='0.35', ha='center', **kw)
    ax.annotate('CILINDRO (D = 1, na origem): não deslizamento por fronteira imersa\n'
                '(gira brevemente para t < 2 para disparar a emissão)\n'
                f'forças: balanço de quantidade de movimento na caixa [{box[0]:g}, {box[1]:g}]²\n'
                '(pontilhado verde: núcleo uniforme da malha, [-1,5; 1,5]²)',
                xy=(0, 0), xytext=(x0 + 0.04 * W, y0 + 0.12 * H), fontsize=9.5,
                arrowprops=dict(arrowstyle='->', color='0.2'))
    ax.annotate('', (0, y0 - 0.1 * H), (x0, y0 - 0.1 * H), arrowprops=dict(arrowstyle='<->', color='0.4'))
    ax.annotate('', (x1, y0 - 0.1 * H), (0, y0 - 0.1 * H), arrowprops=dict(arrowstyle='<->', color='0.4'))
    ax.text(x0 / 2, y0 - 0.14 * H, f'Lu = {-x0:g} D', ha='center', fontsize=9, color='0.4')
    ax.text(x1 / 2, y0 - 0.14 * H, f'Ld = {x1:g} D', ha='center', fontsize=9, color='0.4')
    ax.axvline(0, ymin=0.08, ymax=0.92, color='0.85', lw=0.8, zorder=0)
    ax.annotate('', (x1 + 0.03 * W, y1), (x1 + 0.03 * W, y0), arrowprops=dict(arrowstyle='<->', color='0.4'))
    ax.text(x1 + 0.04 * W, 0, f'2H = {H:g} D\nbloqueio\nD/2H = {100 / H:.2g} %', fontsize=9, color='0.4',
            va='center')
    ax.set_xlim(x0 - 0.12 * W, x1 + 0.14 * W); ax.set_ylim(y0 - 0.18 * H, y1 + 0.1 * H)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Domínio retangular e condições de contorno (em escala; cilindro ampliado se pequeno demais)',
                 fontsize=11)
    fig.savefig(fname, dpi=110, bbox_inches='tight')
    plt.close(fig)


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
        case = CaseDir(OUT, f'cilindro_ret_Re{args.Re:g}_{tag}_h{args.h:.4g}')
        print(f'\n=== Cilindro Re = {args.Re:g}, domínio {tag}: {g.describe()} ===', flush=True)
        plot_domain(g, case.dominio)                 # domínio e malha ANTES de simular
        plot_mesh(g, case.malha)
        case.announce()
        if args.mesh_only:
            continue
        s = CartesianIBNS(g, Cylinder(0.5, pulse), Re=args.Re, cfl=args.cfl)
        n_steps = int(round(args.T / s.dt))
        print(f'  dt = {s.dt:.5f}, {n_steps} passos; nós fantasmas u/v = {len(s.u_rows)}/{len(s.v_rows)}',
              flush=True)
        maybe_resume(args, s, case.parcial)
        t_from = args.stats_from if args.stats_from is not None else args.T / 2
        mon = ConvergenceMonitor(['Cd', 'Cl'], report=args.report,
                                 plot_file=None if args.no_curve else case.convergencia,
                                 live=args.live, refresh=args.refresh, logy=False, ylabel='coeficiente',
                                 dt=s.dt, title=f'Cilindro retangular {tag}')

        def partial(s, case=case):
            plot_case(s, stats(s, min(t_from, s.t / 2)) if s.iter > 10 else None, case.parcial_png)

        s.run(n_steps, monitor=mon, control=control_from_args(args, OUT, case.parcial, on_save=partial))
        res = stats(s, min(t_from, s.t / 2))
        res.update(malha='retangular', domain=tag, Lu=Lu, Ld=Ld, H=H, blockage=0.5 / H, h=args.h,
                   NX=g.NX, NY=g.NY, cells=g.cells, dt=s.dt, t=s.t, steps=s.iter, cfl=s.cfl(),
                   seconds=round(getattr(s, 'seconds_total', 0.0), 1),
                   interrompido=bool(getattr(s, 'stopped', False)))
        results.append(res)
        plot_case(s, res, case.resultado())
        case.save_json(res)
        save_checkpoint(s, case.campos)
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
    est = study_dir(OUT, f'cilindro_ret_Re{args.Re:g}')
    old = []
    if (est / 'resultados.json').exists():          # acumula casos de execuções anteriores
        old = json.load(open(est / 'resultados.json', encoding='utf-8'))
    keep = {(r['domain'], r['h']): r for r in old + results}
    with open(est / 'resultados.json', 'w', encoding='utf-8') as fh:
        json.dump(sorted(keep.values(), key=lambda r: (r['H'], -r['h'])), fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n  abortado pelo usuário (2º Ctrl+C); o último checkpoint (parcial.npz na pasta do caso) foi mantido',
              flush=True)
        raise SystemExit(130)
