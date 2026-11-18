from collections.abc import Mapping

import joblib
import numpy as np
import pandas as pd


def preparar_paciente(paciente, artefato):
    if not isinstance(paciente, Mapping):
        raise ValueError('Informe a paciente como um dicionário.')

    campos = artefato['campos']
    faltantes = sorted(set(campos) - set(paciente))
    extras = sorted(set(paciente) - set(campos))
    if faltantes or extras:
        raise ValueError(f'Campos ausentes: {faltantes}. Campos desconhecidos: {extras}.')

    valores = {}
    for campo in campos:
        valor = paciente[campo]
        if not pd.api.types.is_scalar(valor):
            raise ValueError(f'{campo}: informe apenas um valor.')
        ausente = pd.isna(valor) or str(valor).strip() in ('', '?')
        if ausente:
            if campo not in artefato['campos_anulaveis']:
                raise ValueError(f'{campo}: preenchimento obrigatório.')
            valores[campo] = np.nan
            continue
        valor = str(valor).strip()
        if valor not in artefato['dominios'][campo]:
            raise ValueError(f'{campo}: valor inválido {valor!r}.')
        valores[campo] = valor

    return pd.DataFrame([valores], columns=campos)


def inferir(paciente, caminho_modelo='artefatos/modelo.joblib'):
    artefato = joblib.load(caminho_modelo)
    entrada = preparar_paciente(paciente, artefato)
    classe = str(artefato['pipeline'].predict(entrada)[0])
    return {
        'Class': classe,
        'indicativo': {
            'no-recurrence-events': 'Sem evento de recidiva previsto',
            'recurrence-events': 'Evento de recidiva previsto',
        }[classe],
    }
