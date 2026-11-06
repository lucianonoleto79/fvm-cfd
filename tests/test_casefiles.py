"""Pastas de caso e tabela automática de casos simulados no README."""
import casefiles
from casefiles import CASES_BEGIN, CASES_END, CaseDir, update_readme_cases
from cavity import CavitySolver
from monitor import ConvergenceMonitor


def fake_repo(tmp_path, monkeypatch, markers=True):
    results = tmp_path / 'results'
    results.mkdir()
    readme = tmp_path / 'README.md'
    body = f'# Projeto\n\n## Casos\n\n{CASES_BEGIN}\nvelho\n{CASES_END}\n\n## Depois\n' if markers else '# Projeto\n'
    readme.write_text(body, encoding='utf-8')
    monkeypatch.setattr(casefiles, 'REPO', tmp_path)
    monkeypatch.setattr(casefiles, 'RESULTS', results)
    monkeypatch.setattr(casefiles, 'README', readme)
    return results, readme


def run_case(out):
    s = CavitySolver(Re=100, N=16)
    s.run(max_iter=2000, monitor=ConvergenceMonitor(['u', 'v', 'massa'], report=0))
    case = CaseDir(out, 'cavidade_Re100_N16')
    case.save_json(s.compare_ghia())
    return case


def test_caso_novo_entra_no_readme(tmp_path, monkeypatch):
    results, readme = fake_repo(tmp_path, monkeypatch)
    run_case(results)
    text = readme.read_text(encoding='utf-8')
    assert 'velho' not in text and '## Depois' in text           # só o trecho entre marcadores muda
    assert '[`cavidade_Re100_N16`](results/cavidade_Re100_N16/)' in text
    assert 'Cavidade, Re = 100' in text and '16 × 16' in text and 'convergiu' in text
    first = text
    update_readme_cases(readme, results)                         # idempotente (a menos da data)
    strip = lambda t: '\n'.join(l for l in t.splitlines() if 'atualizado em' not in l)
    assert strip(readme.read_text(encoding='utf-8')) == strip(first)
    assert text.count('cavidade_Re100_N16`]') == 1


def test_caso_fora_de_results_nao_altera_readme(tmp_path, monkeypatch):
    results, readme = fake_repo(tmp_path, monkeypatch)
    before = readme.read_text(encoding='utf-8')
    run_case(tmp_path / 'outra_pasta')
    assert readme.read_text(encoding='utf-8') == before


def test_cria_secao_sem_marcadores(tmp_path, monkeypatch):
    results, readme = fake_repo(tmp_path, monkeypatch, markers=False)
    run_case(results)
    text = readme.read_text(encoding='utf-8')
    assert '## Casos simulados' in text and CASES_BEGIN in text and CASES_END in text
