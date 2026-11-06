"""
Escoamento 2D em torno de um cilindro circular — Re = U D / nu = 100 (padrão).

Malha O (polar) ajustada ao corpo, gerada por polar.PolarGrid: theta uniforme
(NT divisões), r com crescimento geométrico até R_ext, células quase quadradas.
Solver transiente polar.PolarNS (passo fracionário, Crank–Nicolson + Adams–Bashforth 2,
QUICK). Para quebrar a simetria e disparar o desprendimento de vórtices mais cedo, o
cilindro gira brevemente no início (u_t = 0,5 sen(pi t / 2) para t < 2); depois fica parado.

Saídas em results/: malha (cilindro_malha_NT*.png), histórico de Cd e Cl, campo de
vorticidade, métricas (Cd médio, amplitudes, Strouhal) comparadas com a literatura.

Uso:
    python cylinder.py                       # NT = 128 e 192, T = 150
    python cylinder.py --NT 256 --T 150      # uma malha
    python cylinder.py --mesh-only           # só gera e mostra a malha
"""
import argparse
import json
from pathlib import Path

import numpy as np

from control import (add_control_args, clear_stale_commands, control_from_args,
                     maybe_resume, save_checkpoint)
from monitor import ConvergenceMonitor, add_monitor_args
from polar import PolarGrid, PolarNS

DEFAULT_OUT = Path(__file__).resolve().parent.parent / 'results'

# Valores de referência para Re = 100 como costumam ser tabelados na literatura de CFD.
# Confira nas fontes originais antes de citar.
LITERATURE = [
    # fonte, Cd médio, amplitude de Cd, amplitude de Cl, St
    ('Braza, Chassaing & Ha Minh (1986)', 1.364, 0.015, 0.25, 0.160),
    ('Liu, Zheng & Sung (1998)', 1.350, 0.012, 0.339, 0.164),
    ('Park, Kwon & Choi (1998)', 1.33, None, 0.332, 0.165),
    ('Williamson (1988), experimental, St(Re)', None, None, None, 0.164),
]


def pulse(t, amp=0.5, T=2.0):
    return amp * np.sin(np.pi * t / T) if t < T else 0.0


# ------------------------------------------------------------------- figuras
def _lines(ax, g, every_r=1, every_t=1, rmax=None, **kw):
    from matplotlib.collections import LineCollection
    rf = g.rf if rmax is None else g.rf[g.rf <= rmax * 1.0001]
    th = np.linspace(0, 2 * np.pi, 721)
    segs = [np.c_[r * np.cos(th), r * np.sin(th)] for r in rf[::every_r]]
    rr = np.array([rf[0], rf[-1]])
    segs += [np.c_[rr * np.cos(t), rr * np.sin(t)] for t in g.thf[::every_t]]
    ax.add_collection(LineCollection(segs, **kw))


def plot_mesh(g, fname, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
    col = dict(colors='#2b5d8a', linewidths=0.35)
    # (a) domínio inteiro
    a = axs[0]
    _lines(a, g, every_r=4, every_t=4, **col)
    lim = g.Rout * 1.05
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_title(f'(a) domínio inteiro, R_ext = {g.Rout:g} D\n(1 de cada 4 linhas)', fontsize=10)
    a.annotate('escoamento livre U →', (-g.Rout * 0.95, g.Rout * 0.9), fontsize=9)
    # (b) região próxima
    b = axs[1]
    _lines(b, g, rmax=4.0, **col)
    b.set_xlim(-2.2, 2.2); b.set_ylim(-2.2, 2.2)
    b.set_title(f'(b) perto do cilindro (todas as linhas)', fontsize=10)
    # (c) camada junto à parede
    c = axs[2]
    _lines(c, g, rmax=1.0, **dict(col, linewidths=0.6))
    c.set_xlim(-0.75, -0.25); c.set_ylim(-0.25, 0.25)
    c.set_title(f'(c) junto à parede, ponto de estagnação\n'
                f'Δr = {g.drp[0]:.4f} D, r Δθ = {g.R0 * g.dth:.4f} D', fontsize=10)
    for ax in axs:
        ax.add_patch(Circle((0, 0), g.R0, color='0.25', zorder=3))
        ax.set_aspect('equal'); ax.set_xlabel('x / D'); ax.set_ylabel('y / D')
    fig.suptitle(title or f'Malha O: {g.NR} (radial) × {g.NT} (angular) = {g.cells} volumes',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=110, bbox_inches='tight')
    plt.close(fig)


def stats(s, t_from):
    h = np.array(s.history)
    t = h[:, 0] * s.dt
    sel = t >= t_from
    t, cd, cl = t[sel], h[sel, 1], h[sel, 2]
    # cruzamentos ascendentes de Cl por zero (interpolação linear) -> período
    k = np.where((cl[:-1] < 0) & (cl[1:] >= 0))[0]
    tz = t[k] - cl[k] * (t[k + 1] - t[k]) / (cl[k + 1] - cl[k])
    if len(tz) >= 2:
        n = len(tz) - 1
        T = (tz[-1] - tz[0]) / n
        win = (t >= tz[0]) & (t <= tz[-1])           # janela com número inteiro de ciclos
        cdw, clw = cd[win], cl[win]
    else:
        n, T, cdw, clw = 0, np.nan, cd, cl
    return dict(t_from=float(t_from), ciclos=int(n), St=float(1 / T) if n else None,
                Cd_mean=float(cdw.mean()), Cd_amp=float((cdw.max() - cdw.min()) / 2),
                Cl_amp=float((clw.max() - clw.min()) / 2), Cl_rms=float(np.sqrt(np.mean(clw ** 2))),
                Cl_mean=float(clw.mean()))


def plot_case(s, res, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    g = s.g
    h = np.array(s.history)
    t = h[:, 0] * s.dt
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    a = fig.add_subplot(gs[0, 0])
    a.plot(t, h[:, 1], lw=0.9, label='$C_d$')
    a.plot(t, h[:, 2], lw=0.9, label='$C_l$')
    if res:
        a.axvspan(res['t_from'], t[-1], color='0.92', zorder=0, label='janela das médias')
    a.set_ylim(-0.8, 2.2); a.set_xlabel('t U / D'); a.grid(alpha=0.3); a.legend(fontsize=8, ncol=3)
    a.set_title('Coeficientes de força', fontsize=10)
    b = fig.add_subplot(gs[0, 1])
    if res and res['ciclos']:
        w = t >= t[-1] - 3 / res['St']
        b.plot(t[w], h[w, 1], label='$C_d$')
        b2 = b.twinx()
        b2.plot(t[w], h[w, 2], color='C1', label='$C_l$')
        b2.set_ylabel('$C_l$', color='C1'); b.set_ylabel('$C_d$', color='C0')
        b.set_title(f"Últimos 3 ciclos — $\\overline{{C_d}}$ = {res['Cd_mean']:.4f} ± {res['Cd_amp']:.4f}, "
                    f"$C_l$ = ± {res['Cl_amp']:.4f}, St = {res['St']:.4f}", fontsize=10)
    b.set_xlabel('t U / D'); b.grid(alpha=0.3)
    # vorticidade
    c = fig.add_subplot(gs[1, :])
    w = s.vorticity()
    w = np.c_[w, w[:, :1]]
    th = np.r_[g.thf, 2 * np.pi]
    R, T = np.meshgrid(g.rf[1:-1], th, indexing='ij')
    X, Y = R * np.cos(T), R * np.sin(T)
    lv = np.linspace(-3, 3, 25)
    cs = c.contourf(X, Y, np.clip(w, -3, 3), levels=lv, cmap='RdBu_r', extend='both')
    c.add_patch(Circle((0, 0), g.R0, color='0.3', zorder=3))
    c.set_xlim(-2, 20); c.set_ylim(-4, 4); c.set_aspect('equal')
    c.set_xlabel('x / D'); c.set_ylabel('y / D')
    c.set_title(f'Vorticidade $\\omega_z D / U$ em t = {s.t:.1f} (esteira de von Kármán)', fontsize=10)
    fig.colorbar(cs, ax=c, shrink=0.8, pad=0.01)
    fig.suptitle(f'Cilindro, Re = {s.Re:g}, malha {g.NR} × {g.NT}', fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='Cilindro 2D (malha O, transiente) — FVM')
    ap.add_argument('--Re', type=float, default=100)
    ap.add_argument('--NT', type=int, nargs='+', default=[128, 192], help='divisões angulares da malha')
    ap.add_argument('--Rext', type=float, default=40, help='raio do contorno externo, em D')
    ap.add_argument('--aspect', type=float, default=0.7, help='Δr / (r Δθ) das células')
    ap.add_argument('--T', type=float, default=150, help='tempo final, em D/U')
    ap.add_argument('--cfl', type=float, default=0.4, help='CFL do campo inicial (define dt)')
    ap.add_argument('--stats-from', type=float, default=None,
                    help='início da janela das médias (padrão: T/2)')
    ap.add_argument('--mesh-only', action='store_true', help='só gera a figura da malha')
    add_monitor_args(ap, report=1000)
    add_control_args(ap)
    ap.add_argument('--out', default=str(DEFAULT_OUT), help='pasta de saída (padrão: results/)')
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    clear_stale_commands(args, OUT)
    D = 1.0

    results = []
    for NT in args.NT:
        g = PolarGrid(R0=D / 2, Rout=args.Rext * D, NT=NT, aspect=args.aspect)
        print(f'\n=== Cilindro Re = {args.Re:g}: malha {g.describe()} ===', flush=True)
        plot_mesh(g, OUT / f'cilindro_malha_NT{NT}.png')
        print(f'  malha: {OUT / f"cilindro_malha_NT{NT}.png"}', flush=True)
        if args.mesh_only:
            continue
        s = PolarNS(g, Re=args.Re, cfl=args.cfl, wall_speed=pulse)
        n_steps = int(round(args.T / s.dt))
        print(f'  dt = {s.dt:.5f}, {n_steps} passos até t = {args.T:g}', flush=True)
        case = f'cilindro_Re{args.Re:g}_NT{NT}'
        ckpt = OUT / f'{case}_parcial.npz'
        maybe_resume(args, s, ckpt)
        t_from = args.stats_from if args.stats_from is not None else args.T / 2
        mon = ConvergenceMonitor(['Cd', 'Cl'], report=args.report,
                                 plot_file=None if args.no_curve else OUT / f'forcas_{case}.png',
                                 live=args.live, refresh=args.refresh, logy=False,
                                 ylabel='coeficiente', dt=s.dt, title=f'Cilindro Re = {args.Re:g}, NT = {NT}')

        def partial(s, case=case):
            r = stats(s, min(t_from, s.t / 2)) if s.iter > 10 else None
            plot_case(s, r, OUT / f'{case}_parcial.png')

        ctl = control_from_args(args, OUT, ckpt, on_save=partial)
        s.run(n_steps, monitor=mon, control=ctl)
        res = stats(s, min(t_from, s.t / 2))
        res.update(Re=args.Re, NT=NT, NR=g.NR, cells=g.cells, Rext=args.Rext, dt=s.dt, t=s.t,
                   steps=s.iter, cfl=s.cfl(), div_max=float(np.abs(s.divergence()).max()),
                   dr_wall=float(g.drp[0]), seconds=round(getattr(s, 'seconds_total', 0.0), 1),
                   interrompido=bool(getattr(s, 'stopped', False)))
        results.append(res)
        plot_case(s, res, OUT / f'{case}.png')
        save_checkpoint(s, OUT / f'{case}.npz')
        if getattr(s, 'stopped', False):
            print('\n  simulação interrompida: continue com o mesmo comando + --resume', flush=True)
            break

    if not results:
        return
    print(f'\n  NT    NR   volumes   ciclos   Cd médio   amp Cd    amp Cl     St      tempo')
    for r in results:
        st = f"{r['St']:.4f}" if r['St'] else '  -   '
        print(f"{r['NT']:4d} {r['NR']:5d} {r['cells']:8d}   {r['ciclos']:4d}    {r['Cd_mean']:.4f}   "
              f"{r['Cd_amp']:.4f}   {r['Cl_amp']:.4f}   {st}   {r['seconds']:.0f} s"
              + ('  (interrompida)' if r['interrompido'] else ''))
    print('\n  Referências (confira nas fontes):')
    for name, cd, cda, cla, st in LITERATURE:
        f = lambda v, p=3: f'{v:.{p}f}' if v is not None else '  -  '
        print(f'    {name:42s} Cd {f(cd)}  amp Cd {f(cda)}  amp Cl {f(cla)}  St {f(st)}')
    with open(OUT / f'cilindro_Re{args.Re:g}.json', 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n  abortado pelo usuário (2º Ctrl+C); o último checkpoint *_parcial.npz foi mantido',
              flush=True)
        raise SystemExit(130)
