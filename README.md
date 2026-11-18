# Breast Cancer: treinamento e inferência

Projeto acadêmico de classificação do atributo `Class` da base **UCI Breast Cancer (ID 14)**. O alvo indica eventos de recidiva; não classifica tumores como benignos ou malignos.

## Execução

Requer Python 3.11 ou superior. Abra `breast_cancer.ipynb` no Jupyter ou no Google Colab e execute as células em ordem. O notebook é autossuficiente: cria os módulos e baixa os dados se necessário. Para execução local:

```bash
python -m pip install -r requirements.txt
jupyter notebook breast_cancer.ipynb
```

Execute na raiz do projeto. Os dados originais são baixados automaticamente da UCI para `dados/` quando ausentes, com hash SHA-256 verificado no carregamento.

## Módulos

- `treinamento.py`: carrega os dados, separa treino/teste, compara modelos, ajusta hiperparâmetros e salva o melhor pipeline.
- `inferencia.py`: recebe os nove atributos de uma paciente e retorna `Class` e sua interpretação.
- `breast_cancer.ipynb`: execução completa com análise dos dados, comparação, avaliação e exemplo de inferência.

Treinamento sem abrir o notebook:

```python
from treinamento import carregar_dados, treinar

resultado = treinar(carregar_dados())
print(resultado['metricas'])
```

Inferência após o treinamento, inclusive em uma sessão independente:

```python
import json
from pathlib import Path
from inferencia import inferir

paciente = json.loads(Path('paciente_exemplo.json').read_text())
print(inferir(paciente))
```

`node-caps` e `breast-quad` aceitam `None` ou `?`. Todos os nove campos devem existir. Campos extras e categorias inválidas geram `ValueError`.

## Método

- 286 registros, nove atributos e duas classes; valores `?` são ausentes.
- Perfis repetidos são mantidos e agrupados por seus nove atributos. Não há identificador para determinar se representam a mesma paciente.
- Um fold externo de `StratifiedGroupKFold` reserva aproximadamente 20% para teste, sem sobreposição de perfis com o treino.
- `GridSearchCV` com cinco folds estratificados por grupos compara regressão logística, KNN, SVM, árvore de decisão, random forest e Bernoulli Naive Bayes, além do baseline majoritário.
- Imputação pela moda e one-hot são ajustados dentro de cada fold. Não há ajuste do pré-processamento antes da separação.
- A maior acurácia média da validação define o vencedor; em empate, usa-se a ordem da busca. O teste não seleciona o modelo.
- Precisão, recall e F1 da recidiva, acurácia balanceada, relatório por classe e matriz de confusão complementam a acurácia.
- Seed 42. O pipeline persistido é o mesmo avaliado no teste, sem reajuste posterior com o teste.

A avaliação é de um único teste reservado. A média da validação usada na busca não é uma estimativa independente do desempenho.

## Artefatos

`artefatos/comparacao.csv` e `artefatos/metricas.json` registram a execução. `artefatos/modelo.joblib` é gerado ao treinar e inclui o pré-processamento. Use somente arquivos `joblib` confiáveis e as versões de dependências informadas.

Uso acadêmico; este modelo não constitui diagnóstico clínico.

## Fonte dos dados

Zwitter, M. & Soklic, M. (1988). *Breast Cancer*. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C51P4M.

https://archive.ics.uci.edu/dataset/14/breast+cancer

Os arquivos `breast-cancer.data` e `breast-cancer.names` são reproduzidos sem alterações, sob licença CC BY 4.0, com atribuição aos autores. Licença: https://creativecommons.org/licenses/by/4.0/.

## Resultado da execução

| Indicador | Resultado |
| --- | --- |
| Melhor modelo por acurácia na validação | Árvore de decisão |
| Hiperparâmetros | `max_depth=2`, `min_samples_leaf=10` |
| Acurácia média na validação | 76,26% |
| Acurácia no teste reservado | 74,14% (43/58) |
| Recall de recidiva no teste | 23,53% (4/17) |

O recall baixo indica que o modelo deixa de detectar a maioria das recidivas no teste, apesar da acurácia. Os resultados do notebook foram obtidos com as dependências fixadas em `requirements.txt`.
