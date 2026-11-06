"""
Compara o efeito do tamanho do domínio: malha O (cylinder.py, raio R_ext) e domínio
retangular (cylinder_rect.py, [-Lu, Ld] x [-H, H]). Lê os .json dos resultados e gera
tabela + figura.

    python compare_domains.py            # lê results/estudo_cilindro_O_Re100 e _ret_Re100

Saída em results/estudo_dominio_cilindro_Re100/.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from casefiles import study_dir
from cylinder import LITERATURE


def load(root, Re, NT, h):
    """Casos da malha O com NT divisões e do retangular com núcleo h (os demais são outros
    estudos: refinamento de malha)."""
    rows = []
    f = Path(root) / f'estudo_cilindro_O_Re{Re:g}' / 'resultados.json'
    if f.exists():
        for r in json.load(open(f, encoding='utf-8')):
            if r['NT'] == NT:
                rows.append(dict(r, malha='O', L=r['Rext'], label=f"O, R_ext = {r['Rext']:g} D"))
    f = Path(root) / f'estudo_cilindro_ret_Re{Re:g}' / 'resultados.json'
    if f.exists():
        for r in json.load(open(f, encoding='utf-8')):
            if abs(r['h'] - h) < 1e-9:
                rows.append(dict(r, L=r['H'],
                                 label=f"ret. {r['domain']}: [-{r['Lu']:g}, {r['Ld']:g}] × [±{r['H']:g}] D"))
    return sorted(rows, key=lambda r: (r['malha'], r['L']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(Path(__file__).resolve().parent.parent / 'results'))
    ap.add_argument('--Re', type=float, default=100)
    ap.add_argument('--NT', type=int, default=128, help='malha O usada no estudo de domínio')
    ap.add_argument('--h', type=float, default=0.025, help='núcleo do retangular usado no estudo')
    args = ap.parse_args()
    rows = load(args.results, args.Re, args.NT, args.h)
    out = study_dir(args.results, f'dominio_cilindro_Re{args.Re:g}')
    print(f"\n  {'caso':40s} {'volumes':>8s} {'Cd médio':>9s} {'amp Cd':>7s} {'amp Cl':>7s} {'St':>7s}")
    for r in rows:
        print(f"  {r['label']:40s} {r['cells']:8d} {r['Cd_mean']:9.4f} {r['Cd_amp']:7.4f} {r['Cl_amp']:7.4f} "
              f"{r['St']:7.4f}")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for k, (key, name) in enumerate((('Cd_mean', '$\\overline{C_d}$'), ('Cl_amp', 'amplitude de $C_l$'),
                                     ('St', 'St'))):
        a = axs[k]
        for m, mk, lab in (('O', 'o-', 'malha O (raio R_ext)'), ('retangular', 's-', 'retangular (semilargura H)')):
            rr = [r for r in rows if r['malha'] == m]
            if rr:
                a.plot([r['L'] for r in rr], [r[key] for r in rr], mk, label=lab)
        lit = {'Cd_mean': 1, 'Cl_amp': 3, 'St': 4}[key]
        vals = [L[lit] for L in LITERATURE if L[lit] is not None]
        a.axhspan(min(vals), max(vals), color='0.9', zorder=0, label='faixa da literatura')
        a.set_xscale('log'); a.set_xlabel('distância do cilindro ao contorno lateral (R_ext ou H) / D')
        a.set_title(name); a.grid(alpha=0.3, which='both')
        a.set_xticks([10, 20, 40, 80]); a.set_xticklabels(['10', '20', '40', '80'])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0].legend(fontsize=8)
    fig.suptitle('Cilindro, Re = 100 — efeito do tamanho do domínio', fontsize=12)
    fig.tight_layout()
    fig.savefig(out / 'comparacao.png', dpi=120)
    with open(out / 'resultados.json', 'w', encoding='utf-8') as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print(f'\n  figura: {out / "comparacao.png"}')


if __name__ == '__main__':
    main()
