"""
Monitor de convergência para os solvers SIMPLEC (2D e 3D).

Durante a simulação ele
  * imprime os resíduos no terminal em forma de tabela, a cada `report` iterações;
  * desenha a curva de convergência (resíduos x iteração, escala log) e a atualiza a
    cada `refresh` segundos:
      - num arquivo PNG (`plot_file`), regravado de forma atômica, que pode ser aberto
        num visualizador de imagens enquanto o cálculo roda (funciona sem tela, em
        servidores e no GitHub Actions);
      - e/ou numa janela interativa do matplotlib (`live=True`), quando há tela.

Uso:
    mon = ConvergenceMonitor(['u', 'v', 'massa'], tol=1e-6, report=100,
                             plot_file='results/conv.png', title='Cavidade Re = 100')
    solver.run(max_iter=20000, monitor=mon)

O solver chama mon.start(s) antes do laço, mon.update(s) após cada iteração e
mon.finish(s) no fim; os resíduos são lidos de s.history (tuplas (iter, r1, r2, ...)).
"""
import os
import time
from pathlib import Path

import numpy as np

NON_INTERACTIVE = {'agg', 'pdf', 'svg', 'ps', 'cairo', 'pgf', 'template'}


def add_monitor_args(ap, report=100):
    """Opções de linha de comando comuns a cavity.py e pipe.py."""
    g = ap.add_argument_group('monitor de convergência')
    g.add_argument('--report', type=int, default=report,
                   help=f'imprime os resíduos a cada N iterações (0 desliga; padrão {report})')
    g.add_argument('--no-curve', action='store_true',
                   help='não grava a curva de convergência durante a simulação')
    g.add_argument('--live', action='store_true',
                   help='mostra a curva numa janela que se atualiza (precisa de tela)')
    g.add_argument('--refresh', type=float, default=5.0,
                   help='intervalo, em segundos, entre atualizações da curva (padrão 5)')


def monitor_from_args(args, names, tol, plot_file, title):
    return ConvergenceMonitor(names, tol=tol, report=args.report,
                              plot_file=None if args.no_curve else plot_file,
                              live=args.live, refresh=args.refresh, title=title)


class ConvergenceMonitor:
    def __init__(self, names, tol=None, report=100, plot_file=None, live=False,
                 refresh=5.0, title=''):
        self.names = list(names)
        self.tol = tol
        self.report = report
        self.plot_file = Path(plot_file) if plot_file else None
        self.live = live
        self.refresh = refresh
        self.title = title
        self._fig = None
        self._lines = None
        self._last_plot = 0.0
        self._interactive = False

    # ------------------------------------------------------------- terminal
    def _header(self):
        cols = ''.join(f'{n:>11s}' for n in self.names)
        tol = f'   (tol = {self.tol:.0e})' if self.tol else ''
        line = f'  {"iter":>6s}{cols}{"max/tol":>10s}{"it/s":>8s}{"tempo":>9s}'
        print(line + tol, flush=True)
        print('  ' + '-' * (len(line) - 2), flush=True)

    def _row(self, s):
        it, *r = s.history[-1]
        el = time.perf_counter() - self.t0
        rate = (it - self.it0) / el if el > 0 else 0.0
        ratio = f'{max(r) / self.tol:10.2e}' if self.tol else f'{"":10s}'
        cols = ''.join(f'{x:11.3e}' for x in r)
        print(f'  {it:6d}{cols}{ratio}{rate:8.1f}{el:8.1f}s', flush=True)

    # ---------------------------------------------------------------- curva
    def _setup_plot(self):
        if self.live:
            import matplotlib.pyplot as plt
            backend = plt.get_backend().lower()      # sem tela, o pyplot cai em 'agg'
            if backend in NON_INTERACTIVE:
                print(f'  [monitor] backend "{plt.get_backend()}" não é interativo '
                      f'(sem tela?): curva só no arquivo.', flush=True)
            else:
                plt.ion()
                self._fig = plt.figure(figsize=(7, 4.2))
                self._interactive = True
        if self._fig is None:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            self._fig = Figure(figsize=(7, 4.2))
            FigureCanvasAgg(self._fig)
        ax = self._ax = self._fig.add_subplot(111)
        ax.set_yscale('log')
        ax.set_xlim(0, 10)
        ax.set_ylim(min(self.tol or 1e-8, 1e-8), 1.0)   # eixos válidos antes do 1º ponto
        self._lines = [ax.plot([], [], lw=1.2, label=n)[0] for n in self.names]
        if self.tol:
            ax.axhline(self.tol, color='k', ls='--', lw=0.8, label='tolerância')
        ax.set_xlabel('iteração')
        ax.set_ylabel('resíduo')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        self._fig.tight_layout()

    def _draw(self, s, final=False):
        if self._fig is None:
            return
        h = np.asarray(s.history, dtype=float)
        if h.size == 0:
            return
        r = h[:, 1:].copy()
        r[~(r > 0) | ~np.isfinite(r)] = np.nan       # resíduo nulo não cabe em escala log
        for k, ln in enumerate(self._lines):
            ln.set_data(h[:, 0], r[:, k])
        # Escala pelos dados a partir da 2ª iteração: na 1ª o campo inicial é nulo e a
        # normalização do resíduo dá valores sem significado (0 ou ~1e30).
        ref = r[1:] if len(r) > 1 else r
        ax = self._ax
        if np.isfinite(ref).any():
            lo, hi = np.nanmin(ref), np.nanmax(ref)
            if self.tol:
                lo, hi = min(lo, self.tol), max(hi, self.tol)
            ax.set_ylim(lo / 3, hi * 3)
        ax.set_xlim(0, max(h[-1, 0], 10))
        state = ('convergiu' if s.converged else
                 'interrompida' if getattr(s, 'stopped', False) else 'não convergiu') \
            if final else 'em andamento'
        el = time.perf_counter() - self.t0
        ax.set_title(f'{self.title}  —  it {int(h[-1, 0])}, {el:.0f} s, {state}'.strip(' —'),
                     fontsize=9)
        if self.plot_file:
            self.plot_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.plot_file.with_name(self.plot_file.name + '.tmp')
            self._fig.savefig(tmp, dpi=110, format='png')
            os.replace(tmp, self.plot_file)          # quem estiver lendo nunca vê arquivo pela metade
        if self._interactive:
            import matplotlib.pyplot as plt
            self._fig.canvas.draw_idle()
            plt.pause(0.001)
        self._last_plot = time.perf_counter()

    def redraw(self, s):
        """Atualiza a curva agora (ex.: ao pausar ou salvar resultados parciais)."""
        self._draw(s)

    def shift_clock(self, dt):
        """Desconta `dt` segundos (pausa) do tempo e da taxa de iterações."""
        self.t0 += dt
        self._last_plot += dt

    # ------------------------------------------------------ ganchos do solver
    def start(self, s):
        self.t0 = time.perf_counter()
        self.it0 = s.iter
        if self.report:
            self._header()
        if self.plot_file or self.live:
            self._setup_plot()
            self._last_plot = self.t0

    def update(self, s):
        if self.report and (s.iter % self.report == 0 or s.converged):
            self._row(s)
        if self._fig is not None and time.perf_counter() - self._last_plot >= self.refresh:
            self._draw(s)

    def finish(self, s):
        if self.report and s.history and s.iter % self.report != 0 and not s.converged:
            self._row(s)                             # última linha quando parou por max_iter
        if self.report:
            msg = ('convergiu' if s.converged else
                   'interrompida pelo usuário' if getattr(s, 'stopped', False) else 'NÃO convergiu')
            print(f'  -> {msg} em {s.iter} iterações, {time.perf_counter() - self.t0:.1f} s',
                  flush=True)
        self._draw(s, final=True)
        if self._interactive:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.close(self._fig)
