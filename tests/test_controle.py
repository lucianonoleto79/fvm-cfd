"""Testes do controle da simulação: checkpoint/retomada, pausar, salvar parcial, interromper."""
import os
import signal
import threading
import time

import numpy as np
import pytest

from cavity import CavitySolver
from control import RunControl, load_checkpoint, save_checkpoint
from monitor import ConvergenceMonitor


def quiet():
    return ConvergenceMonitor(['u', 'v', 'massa'], report=0)


def ctl(tmp_path, **kw):
    kw.setdefault('checkpoint_file', tmp_path / 'ck.npz')
    kw.setdefault('checkpoint_every', 0)
    return RunControl(control_dir=tmp_path, keyboard=False, verbose=False, **kw)


def test_retomada_reproduz_rodada_continua(tmp_path):
    ref = CavitySolver(Re=100, N=16)
    ref.run(max_iter=2000, monitor=quiet())

    a = CavitySolver(Re=100, N=16)
    a.run(max_iter=60, monitor=quiet(), control=ctl(tmp_path))     # para em max_iter
    assert not a.converged and (tmp_path / 'ck.npz').exists()

    b = load_checkpoint(CavitySolver(Re=100, N=16), tmp_path / 'ck.npz')
    assert b.iter == 60 and len(b.history) == 60
    b.run(max_iter=2000, monitor=quiet(), control=ctl(tmp_path))
    assert b.converged and b.iter == ref.iter
    np.testing.assert_allclose(b.u, ref.u, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.array(b.history), np.array(ref.history), rtol=1e-9, atol=1e-14)
    assert not (tmp_path / 'ck.npz').exists()                       # convergiu: checkpoint apagado


def test_checkpoint_recusa_malha_diferente(tmp_path):
    s = CavitySolver(Re=100, N=16)
    s.run(max_iter=5, monitor=quiet())
    save_checkpoint(s, tmp_path / 'ck.npz')
    with pytest.raises(ValueError):
        load_checkpoint(CavitySolver(Re=100, N=32), tmp_path / 'ck.npz')


def test_arquivo_parar_interrompe_e_salva(tmp_path):
    (tmp_path / 'PARAR').touch()
    s = CavitySolver(Re=100, N=16)
    s.run(max_iter=2000, monitor=quiet(), control=ctl(tmp_path))
    assert s.stopped and not s.converged and s.iter == 1
    assert not (tmp_path / 'PARAR').exists()                        # comando consumido
    assert int(np.load(tmp_path / 'ck.npz')['iter']) == 1


def test_arquivo_salvar_grava_parcial(tmp_path):
    (tmp_path / 'SALVAR').touch()
    calls = []
    s = CavitySolver(Re=100, N=16)
    s.run(max_iter=2000, monitor=quiet(), control=ctl(tmp_path, on_save=lambda s: calls.append(s.iter)))
    assert calls == [1] and s.converged and not (tmp_path / 'SALVAR').exists()


def test_arquivo_pausar(tmp_path):
    (tmp_path / 'PAUSAR').touch()
    threading.Timer(0.6, (tmp_path / 'PAUSAR').unlink).start()
    c = ctl(tmp_path)
    s = CavitySolver(Re=100, N=16)
    t0 = time.perf_counter()
    s.run(max_iter=2000, monitor=quiet(), control=c)
    wall = time.perf_counter() - t0
    assert s.converged and c.paused_total >= 0.5
    assert s.seconds <= wall - 0.5 + 0.05                           # pausa fora do tempo de cálculo


@pytest.mark.skipif(os.name == 'nt', reason='no Windows os.kill(SIGINT) encerra o processo')
def test_ctrl_c_interrompe_com_elegancia(tmp_path):
    c = ctl(tmp_path)
    s = CavitySolver(Re=1000, N=32)                                 # demora o bastante
    threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGINT)).start()
    s.run(max_iter=20000, monitor=quiet(), control=c)
    assert s.stopped and not s.converged and s.iter > 0
    assert int(np.load(tmp_path / 'ck.npz')['iter']) == s.iter
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler   # handler restaurado
