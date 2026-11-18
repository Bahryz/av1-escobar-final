from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import json

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SEED = 42
TARGET = 'Class'
CLASSES = ['no-recurrence-events', 'recurrence-events']
URL = 'https://archive.ics.uci.edu/static/public/14/breast+cancer.zip'
DATA_SHA256 = 'ca7d3fa97b62ff967b6894ffbb3acaaf4a51062e0149e5950b3ad6f685970b65'
DOMINIOS = {
    'age': [f'{i}-{i + 9}' for i in range(10, 100, 10)],
    'menopause': ['lt40', 'ge40', 'premeno'],
    'tumor-size': [f'{i}-{i + 4}' for i in range(0, 60, 5)],
    'inv-nodes': [f'{i}-{i + 2}' for i in range(0, 36, 3)] + ['36-39'],
    'node-caps': ['yes', 'no'],
    'deg-malig': ['1', '2', '3'],
    'breast': ['left', 'right'],
    'breast-quad': ['left-up', 'left-low', 'right-up', 'right-low', 'central'],
    'irradiat': ['yes', 'no'],
}
CAMPOS = list(DOMINIOS)


def carregar_dados(caminho='dados/breast-cancer.data'):
    caminho = Path(caminho)
    if caminho.exists():
        conteudo = caminho.read_bytes()
    else:
        with urlopen(URL, timeout=60) as resposta:
            with ZipFile(BytesIO(resposta.read())) as arquivo:
                conteudo = arquivo.read('breast-cancer.data')
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_bytes(conteudo)

    if sha256(conteudo).hexdigest() != DATA_SHA256:
        raise ValueError('O arquivo não corresponde à base Breast Cancer UCI 14.')

    return pd.read_csv(
        BytesIO(conteudo), names=[TARGET, *CAMPOS], na_values='?', dtype=str,
    )


def definir_busca():
    return [
        {'modelo': [DummyClassifier(strategy='most_frequent')]},
        {
            'modelo': [LogisticRegression(max_iter=2000, random_state=SEED)],
            'modelo__C': [0.01, 0.1, 1, 10],
        },
        {
            'modelo': [KNeighborsClassifier()],
            'modelo__n_neighbors': [3, 5, 9, 15],
            'modelo__weights': ['uniform', 'distance'],
        },
        {
            'modelo': [SVC(random_state=SEED)],
            'modelo__C': [0.1, 1, 10],
            'modelo__kernel': ['linear', 'rbf'],
        },
        {
            'modelo': [DecisionTreeClassifier(random_state=SEED)],
            'modelo__max_depth': [2, 4, None],
            'modelo__min_samples_leaf': [2, 5, 10],
        },
        {
            'modelo': [RandomForestClassifier(n_estimators=200, random_state=SEED)],
            'modelo__max_depth': [3, 6, None],
            'modelo__min_samples_leaf': [2, 5],
        },
        {'modelo': [BernoulliNB()], 'modelo__alpha': [0.1, 1, 10]},
    ]


def treinar(dados, pasta_saida='artefatos'):
    X, y = dados[CAMPOS], dados[TARGET]
    grupos = pd.util.hash_pandas_object(X.fillna('?'), index=False)
    divisao = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    treino, teste = next(divisao.split(X, y, groups=grupos))
    X_treino, X_teste = X.iloc[treino], X.iloc[teste]
    y_treino, y_teste = y.iloc[treino], y.iloc[teste]

    pipeline = Pipeline([
        ('imputacao', SimpleImputer(strategy='most_frequent')),
        ('codificacao', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('modelo', DummyClassifier()),
    ])
    validacao = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    busca = GridSearchCV(
        pipeline, definir_busca(), scoring='accuracy', cv=validacao,
        n_jobs=1, refit=True, error_score='raise',
    )
    busca.fit(X_treino, y_treino, groups=grupos.iloc[treino])

    resultados = pd.DataFrame(busca.cv_results_)
    resultados['modelo'] = [
        type(parametros['modelo']).__name__ for parametros in resultados['params']
    ]
    comparacao = (
        resultados.sort_values('rank_test_score', kind='stable')
        .drop_duplicates('modelo')
        .rename(columns={
            'mean_test_score': 'acuracia_cv',
            'std_test_score': 'desvio_cv',
            'rank_test_score': 'posicao',
        })[['modelo', 'acuracia_cv', 'desvio_cv', 'posicao']]
        .reset_index(drop=True)
    )

    modelo = busca.best_estimator_
    predicoes = modelo.predict(X_teste)
    parametros = {
        chave.removeprefix('modelo__'): valor
        for chave, valor in busca.best_params_.items() if chave != 'modelo'
    }
    metricas = {
        'modelo': type(modelo.named_steps['modelo']).__name__,
        'parametros': parametros,
        'acuracia_cv': float(busca.best_score_),
        'acuracia_teste': accuracy_score(y_teste, predicoes),
        'acuracia_balanceada_teste': balanced_accuracy_score(y_teste, predicoes),
        'precisao_recidiva': precision_score(
            y_teste, predicoes, pos_label=CLASSES[1], zero_division=0,
        ),
        'recall_recidiva': recall_score(
            y_teste, predicoes, pos_label=CLASSES[1], zero_division=0,
        ),
        'f1_recidiva': f1_score(
            y_teste, predicoes, pos_label=CLASSES[1], zero_division=0,
        ),
        'n_treino': len(treino),
        'n_teste': len(teste),
        'seed': SEED,
        'sklearn': sklearn.__version__,
        'relatorio': classification_report(
            y_teste, predicoes, labels=CLASSES, output_dict=True, zero_division=0,
        ),
    }
    artefato = {
        'pipeline': modelo, 'campos': CAMPOS, 'dominios': DOMINIOS,
        'campos_anulaveis': ['node-caps', 'breast-quad'],
        'classes': CLASSES, 'metricas': metricas,
    }
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefato, pasta / 'modelo.joblib')
    comparacao.to_csv(pasta / 'comparacao.csv', index=False)
    (pasta / 'metricas.json').write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    return {
        'artefato': artefato, 'comparacao': comparacao, 'metricas': metricas,
        'matriz_confusao': confusion_matrix(y_teste, predicoes, labels=CLASSES),
        'indices_treino': treino, 'indices_teste': teste,
    }
