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
"""
import json
from pathlib import Path


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

    def save_json(self, data):
        with open(self.json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def announce(self):
        print(f'  pasta do caso: {self.root}', flush=True)
        print(f'  domínio: {self.dominio.name}   malha: {self.malha.name}   '
              f'(convergência em {self.convergencia.name} durante a simulação)', flush=True)


def study_dir(out, name):
    d = Path(out) / f'estudo_{name}'
    d.mkdir(parents=True, exist_ok=True)
    return d
