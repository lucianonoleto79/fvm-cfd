"""
Organização padronizada dos resultados: uma pasta por caso simulado.

    results/<caso>/
        01_dominio.png        domínio e condições de contorno   (gerado ANTES de simular)
        02_malha.png          malha                              (gerado ANTES de simular)
        03_convergencia.png   curva de convergência / histórico  (atualizada DURANTE a simulação)
        04_resultados*.png    figuras dos resultados             (gerado no fim)
        resultados.json       métricas
        campos.npz            campos finais (u, v, p, histórico ...; não versionado no git)
        parcial.npz           checkpoint para --resume (casos permanentes: só enquanto não
                              convergiram; transientes: fica, para estender a simulação)
        05_animacao_*.gif     animação (cilindro, opção --animate)
        simulacao.log         saída do terminal (quando a simulação roda em segundo plano)
        parcial.png           figura do estado parcial (ao pedir "salvar parcial")

Estudos que combinam vários casos (estudo de malha, de domínio) ficam em
results/estudo_<nome>/.

Ao gravar resultados.json de um caso em results/ do repositório, a tabela de casos
simulados do README.md (entre os marcadores CASES_BEGIN e CASES_END) é regenerada a partir
de todas as pastas de caso — assim todo caso novo entra no README automaticamente. Para
regenerar à mão:  python python/casefiles.py
"""
import datetime
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / 'results'
README = REPO / 'README.md'
CASES_BEGIN = '<!-- casos:inicio (tabela gerada por python/casefiles.py; não edite à mão) -->'
CASES_END = '<!-- casos:fim -->'


class CaseDir:
    def __init__(self, out, name):
        self.root = Path(out) / name
        self.name = name
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def dominio(self):
        return self.root / '01_dominio.png'

    @property
    def malha(self):
        return self.root / '02_malha.png'

    @property
    def convergencia(self):
        return self.root / '03_convergencia.png'

    def resultado(self, suffix=''):
        return self.root / f'04_resultados{("_" + suffix) if suffix else ""}.png'

    @property
    def json(self):
        return self.root / 'resultados.json'

    @property
    def campos(self):
        return self.root / 'campos.npz'

    @property
    def parcial(self):
        return self.root / 'parcial.npz'

    @property
    def parcial_png(self):
        return self.root / 'parcial.png'

    def save_json(self, data, update_readme=True):
        data = dict(data)
        data.setdefault('data', datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
        with open(self.json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # só os casos da pasta results/ do repositório entram no README
        if update_readme and self.root.parent.resolve() == RESULTS.resolve() and README.exists():
            n = update_readme_cases()
            print(f'  README.md: tabela de casos simulados atualizada ({n} casos)', flush=True)

    def announce(self):
        print(f'  pasta do caso: {self.root}', flush=True)
        print(f'  domínio: {self.dominio.name}   malha: {self.malha.name}   '
              f'(convergência em {self.convergencia.name} durante a simulação)', flush=True)


def study_dir(out, name):
    d = Path(out) / f'estudo_{name}'
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------- tabela no README
def _fmt(v, p=4):
    return '—' if v is None else f'{v:.{p}f}'.replace('-', '−').replace('.', ',')


def _int(n):
    return f'{n:,}'.replace(',', '.')                          # 137676 -> 137.676


def _tempo(s):
    if not s:
        return '—'
    return f'{s:.0f} s' if s < 60 else f'{s / 60:.0f} min' if s < 5400 else f'{s / 3600:.1f} h'.replace('.', ',')


def describe_case(name, r):
    """(tipo, malha, métricas principais, situação) de um caso a partir de resultados.json."""
    if r.get('interrompido'):
        status = 'interrompido (parcial)'
    elif 'Cd_mean' in r or 'lago' in r:
        status = 'concluído'
    else:
        status = 'convergiu' if r.get('converged') else 'não convergiu'
    if 'psi_err_pct' in r:                                     # cavidade
        tipo = f"Cavidade, Re = {r['Re']}"
        malha = f"{r['N']} × {r['N']}"
        met = (f"ψ_min = {_fmt(r['vortex']['psi'], 5)} (Ghia {_fmt(r['ghia_vortex']['psi'], 5)}, "
               f"erro {_fmt(r['psi_err_pct'], 2)} %); {r['iter']} iterações")
    elif 'fRe' in r:                                           # tubo 3D
        m = re.search(r'Re(\d+(?:\.\d+)?)', name)
        tipo = f"Tubo 3D, Re = {r.get('Re', m.group(1) if m else '?')}"
        malha = f"{r['n']} células no D ({_int(r['cells'])} volumes)"
        met = (f"f·Re = {_fmt(r['fRe'], 2)} (erro {_fmt(r['fRe_err_pct'], 2)} %); "
               f"L/D = {_fmt(r['Le'], 2)} (Durst {_fmt(r['Le_durst'], 2)})")
    elif 'Cd_mean' in r:                                       # cilindro
        if 'NT' in r:
            tipo = f"Cilindro, malha O, Re = {r['Re']:g}, R_ext = {r['Rext']:g} D"
            malha = f"{r['NR']} × {r['NT']} ({_int(r['cells'])} volumes)"
        else:
            tipo = (f"Cilindro, retangular {r['domain']}, Re = {r.get('Re', 100):g}, "
                    f"[−{r['Lu']:g}, {r['Ld']:g}] × [±{r['H']:g}] D")
            malha = f"{r['NX']} × {r['NY']}, h = D/{1 / r['h']:.0f} ({_int(r['cells'])} volumes)"
        met = (f"Cd = {_fmt(r['Cd_mean'])}, Cl = ±{_fmt(r['Cl_amp'])}, St = {_fmt(r.get('St'))} "
               f"({r.get('ciclos', 0)} ciclos)")
    elif 'lago' in r:                                          # lago com escalar (paranoa.py)
        tipo = f"Lago {r['lago']}: poluente (escalar), {r['dias']:g} dias (ilustrativo)"
        malha = f"{r['NX']} × {r['NY']}, h = {r['h']:g} m ({_int(r['cells'])} no lago)"
        rf = r['sondas'].get('Braço do Riacho Fundo')
        met = (f"C/C0 média = {_fmt(r['C_media'], 4)}; sonda do braço do Riacho Fundo {_fmt(rf, 4)}; "
               f"Q na barragem {_fmt(r['Q_saida'], 2)} m³/s; massa fecha a {r['balanco_rel']:.0e}")
    else:
        tipo, malha, met = name, '—', '—'
    return tipo, malha, met, status


def cases_table(results=None):
    results = RESULTS if results is None else results
    rows = []
    for f in sorted(Path(results).glob('*/resultados.json')):
        name = f.parent.name
        if name.startswith('estudo_'):
            continue
        r = json.load(open(f, encoding='utf-8'))
        tipo, malha, met, status = describe_case(name, r)
        rel = f.parent.relative_to(REPO).as_posix() if f.parent.is_relative_to(REPO) else f'results/{name}'
        figs = ' · '.join(f'[{lab}]({rel}/{fn})' for lab, fn in
                          (('domínio', '01_dominio.png'), ('malha', '02_malha.png'),
                           ('convergência', '03_convergencia.png'), ('resultados', '04_resultados.png'))
                          if (f.parent / fn).exists())
        tempo = _tempo(r.get('seconds'))
        rows.append(f"| [`{name}`]({rel}/) | {tipo} | {malha} | {met} | {status} | {tempo} | "
                    f"{r.get('data', '—')} | {figs} |")
    head = ('| caso | tipo | malha | resultados principais | situação | tempo | data | figuras |\n'
            '|---|---|---|---|---|---:|---|---|')
    return head + '\n' + '\n'.join(rows), len(rows)


def update_readme_cases(readme=None, results=None):
    """Reescreve a tabela de casos entre os marcadores (cria a seção no fim se não existir)."""
    readme = README if readme is None else readme
    results = RESULTS if results is None else results
    table, n = cases_table(results)
    block = (f'{CASES_BEGIN}\n_{n} casos em `results/`, atualizado em '
             f'{datetime.datetime.now():%Y-%m-%d %H:%M}._\n\n{table}\n{CASES_END}')
    text = Path(readme).read_text(encoding='utf-8')
    if CASES_BEGIN in text and CASES_END in text:
        a = text.index(CASES_BEGIN)
        b = text.index(CASES_END) + len(CASES_END)
        text = text[:a] + block + text[b:]
    else:
        text = text.rstrip('\n') + '\n\n## Casos simulados\n\n' + block + '\n'
    Path(readme).write_text(text, encoding='utf-8')
    return n


if __name__ == '__main__':
    print(f'README.md: {update_readme_cases()} casos na tabela')
