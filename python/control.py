"""
Controle da simulação pelo usuário: salvar resultados parciais, pausar e interromper.

Três formas de comandar uma simulação em andamento:

  Teclado (terminal interativo; Linux, macOS e Windows):
      p  pausa / continua
      s  salva resultados parciais agora (checkpoint + figuras parciais)
      q  interrompe: termina a iteração atual, salva e encerra com elegância
  Ctrl+C
      1ª vez: igual a "q";  2ª vez: aborta imediatamente (fica o último checkpoint)
  Arquivos na pasta de controle (padrão: a pasta de saída, results/) — úteis com a
  simulação em segundo plano, por ssh, no Codespaces ou no GitHub Actions:
      PAUSAR   pausa enquanto o arquivo existir
      SALVAR   salva resultados parciais (o arquivo é apagado em seguida)
      PARAR    interrompe (o arquivo é apagado em seguida)
  ex.: `touch results/PAUSAR` … `rm results/PAUSAR`;  no PowerShell: `ni results/PARAR`.
  Os arquivos valem para todas as simulações que usam a mesma pasta.

Além disso grava um checkpoint a cada `checkpoint_every` segundos. O checkpoint é um
.npz com o estado do solver (campos, histórico de resíduos, iteração) e permite
retomar o cálculo com `--resume`. Ele é gravado de forma atômica: uma interrupção
durante a escrita não corrompe o checkpoint anterior.

O solver precisa expor get_state() -> dict de arrays e set_state(dict).
"""
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

KEYS_HELP = 'teclas: [p] pausa/continua  [s] salva parcial  [q] interrompe  (Ctrl+C = q)'


# ----------------------------------------------------------------- checkpoints
def save_checkpoint(s, fname):
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    state = s.get_state()
    state.update(history=np.asarray(s.history, dtype=float).reshape(len(s.history), -1),
                 iter=s.iter, converged=s.converged,
                 seconds=getattr(s, 'seconds_total', 0.0))
    tmp = fname.with_name(fname.name + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez(f, **state)
    os.replace(tmp, fname)


def load_checkpoint(s, fname):
    d = np.load(fname)
    s.set_state({k: d[k] for k in d.files
                 if k not in ('history', 'iter', 'converged', 'seconds')})
    s.history = [(int(r[0]), *r[1:]) for r in d['history'].tolist()]
    s.iter = int(d['iter'])
    s.converged = bool(d['converged'])
    s.seconds_total = float(d['seconds'])
    return s


# ------------------------------------------------------------ linha de comando
def add_control_args(ap):
    """Opções comuns a cavity.py e pipe.py."""
    g = ap.add_argument_group('controle da simulação (pausar / salvar parciais / interromper)')
    g.add_argument('--checkpoint', type=float, default=300.0, metavar='S',
                   help='grava checkpoint a cada S segundos (0: só ao pedir/interromper; padrão 300)')
    g.add_argument('--resume', action='store_true',
                   help='retoma cada caso do checkpoint *_parcial.npz, se existir')
    g.add_argument('--no-keys', action='store_true',
                   help='desliga os comandos pelo teclado (p/s/q)')
    g.add_argument('--control-dir', default=None,
                   help='pasta vigiada pelos arquivos PAUSAR/SALVAR/PARAR (padrão: a de --out)')


def clear_stale_commands(args, out):
    """No início do script: remove PARAR/SALVAR que sobraram de uma execução anterior
    (criados depois que ela já tinha terminado). PAUSAR é mantido: pausa de propósito."""
    d = Path(args.control_dir or out)
    for name in ('PARAR', 'SALVAR'):
        f = d / name
        if f.exists():
            f.unlink()
            print(f'  [controle] arquivo de comando antigo removido: {f}', flush=True)


def control_from_args(args, out, checkpoint_file, on_save=None):
    return RunControl(checkpoint_file=checkpoint_file, checkpoint_every=args.checkpoint,
                      control_dir=args.control_dir or out, keyboard=not args.no_keys,
                      on_save=on_save)


def maybe_resume(args, s, checkpoint_file, final_file=None):
    """Com --resume: continua do checkpoint; sem checkpoint, reaproveita um caso já
    concluído (final_file gravado por save_checkpoint); senão começa do zero."""
    if not args.resume:
        return False
    if Path(checkpoint_file).exists():
        load_checkpoint(s, checkpoint_file)
        print(f'  retomando de {checkpoint_file} (it {s.iter}, '
              f'{s.seconds_total:.0f} s já calculados)', flush=True)
        return True
    if final_file is not None and Path(final_file).exists():
        with np.load(final_file) as d:
            complete = {'iter', 'converged'} <= set(d.files) and bool(d['converged'])
        if complete:
            load_checkpoint(s, final_file)
            print(f'  caso já concluído em {final_file} (it {s.iter}): reaproveitado', flush=True)
            return True
    print(f'  sem checkpoint em {checkpoint_file}: começando do zero', flush=True)
    return False


# --------------------------------------------------------------------- teclado
class _Keyboard:
    """Leitura não bloqueante de uma tecla, sem precisar de Enter."""

    def __init__(self):
        self.ok = False
        self._old = None
        try:
            if not sys.stdin or not sys.stdin.isatty():
                return
        except (AttributeError, ValueError):
            return
        if os.name == 'nt':
            import msvcrt
            self._msvcrt = msvcrt
            self.ok = True
        else:
            import termios
            import tty
            try:
                self._fd = sys.stdin.fileno()
                self._old = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)              # mantém Ctrl+C funcionando
                self.ok = True
            except (termios.error, OSError):
                self._old = None

    def get(self):
        if not self.ok:
            return None
        if os.name == 'nt':
            if self._msvcrt.kbhit():
                return self._msvcrt.getwch().lower()
            return None
        import select
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            ch = os.read(self._fd, 1)
            return ch.decode(errors='ignore').lower() or None
        return None

    def close(self):
        if self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._old = None


# -------------------------------------------------------------------- controle
class RunControl:
    def __init__(self, checkpoint_file=None, checkpoint_every=300.0, control_dir=None,
                 keyboard=True, on_save=None, verbose=True):
        """
        checkpoint_file   .npz do checkpoint (None: não grava checkpoints)
        checkpoint_every  intervalo em segundos entre checkpoints automáticos (0: só a pedido)
        control_dir       pasta vigiada pelos arquivos PAUSAR / SALVAR / PARAR (None: desliga)
        keyboard          aceita comandos pelo teclado quando há terminal interativo
        on_save(s)        chamada extra ao salvar resultados parciais (ex.: figuras do campo)
        """
        self.checkpoint_file = Path(checkpoint_file) if checkpoint_file else None
        self.checkpoint_every = checkpoint_every
        self.control_dir = Path(control_dir) if control_dir else None
        self.use_keyboard = keyboard
        self.on_save = on_save
        self.verbose = verbose
        self.stop_requested = False
        self.paused_total = 0.0
        self._sigint = 0

    def _say(self, msg):
        if self.verbose:
            print(f'  [controle] {msg}', flush=True)

    def _file(self, name):
        return self.control_dir / name if self.control_dir else None

    def _take(self, name):
        """True se o arquivo de comando existe; apaga-o (comandos de disparo único)."""
        f = self._file(name)
        if f is not None and f.exists():
            try:
                f.unlink()
            except OSError:
                pass
            return True
        return False

    def _paused_by_file(self):
        f = self._file('PAUSAR')
        return f is not None and f.exists()

    # --------------------------------------------------------------- ciclo
    def start(self, s):
        self._kb = _Keyboard() if self.use_keyboard else None
        self._last_ckpt = time.perf_counter()
        self._old_handler = None
        if threading.current_thread() is threading.main_thread():
            self._old_handler = signal.signal(signal.SIGINT, self._on_sigint)
        if self.verbose:
            hints = []
            if self._kb is not None and self._kb.ok:
                hints.append(KEYS_HELP)
            else:
                hints.append('Ctrl+C interrompe e salva')
            if self.control_dir:
                hints.append(f'arquivos PAUSAR/SALVAR/PARAR em {self.control_dir}')
            self._say(';  '.join(hints))

    def _on_sigint(self, signum, frame):
        self._sigint += 1
        if self._sigint >= 2:
            raise KeyboardInterrupt
        self.stop_requested = True
        print('\n  [controle] Ctrl+C: interrompendo ao fim desta iteração '
              '(Ctrl+C de novo aborta sem salvar)', flush=True)

    def save(self, s, monitor=None, why='a pedido'):
        if self.checkpoint_file:
            save_checkpoint(s, self.checkpoint_file)
        if monitor is not None:
            monitor.redraw(s)
        if self.on_save is not None:
            try:
                self.on_save(s)
            except Exception as e:                   # figura parcial falhou: não derruba o cálculo
                self._say(f'aviso: falha ao gerar resultados parciais extras: {e!r}')
        self._last_ckpt = time.perf_counter()
        where = f' em {self.checkpoint_file}' if self.checkpoint_file else ''
        self._say(f'resultados parciais salvos ({why}, it {s.iter}){where}')

    def _pause(self, s, monitor, by_file):
        t0 = time.perf_counter()
        if by_file:
            self._say(f'PAUSADO na it {s.iter} — apague {self._file("PAUSAR")} para continuar')
        else:
            self._say(f'PAUSADO na it {s.iter} — [p] continua, [s] salva, [q] interrompe')
        if monitor is not None:
            monitor.redraw(s)
        while True:
            time.sleep(0.2)
            key = self._kb.get() if self._kb else None
            if key == 's' or self._take('SALVAR'):
                self.save(s, monitor)
            if key == 'q' or self._take('PARAR') or self.stop_requested:
                self.stop_requested = True
                break
            if by_file and not self._paused_by_file():
                break
            if not by_file and key == 'p':
                break
        dt = time.perf_counter() - t0
        self.paused_total += dt
        if monitor is not None:
            monitor.shift_clock(dt)                  # pausa não conta como tempo de cálculo
        self._last_ckpt += dt
        if not self.stop_requested:
            self._say(f'continuando (pausa de {dt:.0f} s)')

    def poll(self, s, monitor=None):
        """Chamado após cada iteração. Devolve True se a simulação deve parar."""
        key = self._kb.get() if self._kb else None
        if key == 'q' or self._take('PARAR'):
            self.stop_requested = True
        if not self.stop_requested:
            if key == 's' or self._take('SALVAR'):
                self.save(s, monitor)
            if key == 'p' or self._paused_by_file():
                self._pause(s, monitor, by_file=key != 'p')
        if (not self.stop_requested and self.checkpoint_every and self.checkpoint_file
                and time.perf_counter() - self._last_ckpt >= self.checkpoint_every):
            save_checkpoint(s, self.checkpoint_file)
            self._last_ckpt = time.perf_counter()
        if self.stop_requested:
            self._say(f'interrompida pelo usuário na it {s.iter}')
        return self.stop_requested

    def finish(self, s, aborted=False):
        if self._old_handler is not None:
            signal.signal(signal.SIGINT, self._old_handler)
            self._old_handler = None
        if self._kb is not None:
            self._kb.close()                         # devolve o terminal ao modo normal
        if aborted:
            # aborto no meio de uma iteração: os campos podem estar pela metade;
            # preserva o último checkpoint completo
            if self.checkpoint_file and self.checkpoint_file.exists():
                self._say(f'cálculo abortado (Ctrl+C duplo ou erro); último checkpoint '
                          f'mantido em {self.checkpoint_file}')
            return
        if self.checkpoint_file:
            if s.converged:
                # convergiu: o checkpoint não serve mais para retomar
                if self.checkpoint_file.exists():
                    self.checkpoint_file.unlink()
            else:                                    # o script gera as figuras finais
                save_checkpoint(s, self.checkpoint_file)
                self._say(f'checkpoint da it {s.iter} em {self.checkpoint_file} '
                          f'(retome com --resume)')


# ----------------------------------------------------------------------- laço
def run_loop(s, max_iter, monitor, control=None):
    """Laço SIMPLEC comum aos solvers 2D e 3D (s.iterate() faz uma iteração)."""
    s.stopped = False
    s.seconds_total = getattr(s, 'seconds_total', 0.0)
    t0 = time.perf_counter()
    monitor.start(s)
    if control is not None:
        control.start(s)
    aborted = True
    try:
        while not s.converged and s.iter < max_iter:
            s.iterate()
            monitor.update(s)
            if control is not None and control.poll(s, monitor):
                s.stopped = True
                break
        aborted = False
    finally:
        paused = control.paused_total if control is not None else 0.0
        s.seconds = time.perf_counter() - t0 - paused
        s.seconds_total += s.seconds
        if control is not None:
            control.finish(s, aborted=aborted)
    monitor.finish(s)
    return s.converged
