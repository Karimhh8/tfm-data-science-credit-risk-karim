"""
src/transformers.py
===================
Transformers personalizados compartidos entre notebooks.

Este módulo contiene las clases que extienden scikit-learn para el preprocesamiento específico del dataset HMEQ en el marco del TFM de riesgo de crédito.

Importar desde cualquier notebook con:
    from src.transformers import CLTVTransformer
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CLTVTransformer(BaseEstimator, TransformerMixin):
    """
    Construye la variable CLTV (Combined Loan-to-Value) como ratio financiero:

        CLTV = (LOAN + MORTDUE) / VALUE

    A diferencia del LTV clásico, CLTV incluye tanto el préstamo solicitado (LOAN) como el principal pendiente de la hipoteca actual (MORTDUE), lo que refleja correctamente la exposición total al valor de la garantía
    en un préstamo Home Equity.

    Posteriormente aplica un cap económico (2.55 por defecto) sobre el CLTV resultante. El umbral se deriva del análisis de outliers del EDA: valores superiores corresponden a registros con VALUE implausible y no se
    consideran económicamente realistas.

    Elimina MORTDUE y VALUE del DataFrame, ya que la información queda capturada por CLTV — evita redundancia informativa y multicolinealidad.

    Parameters
    ----------
    cap : float, default=2.55
        Umbral superior aplicado a CLTV tras su construcción.
    """

    def __init__(self, cap=2.55):
        self.cap = cap

    def fit(self, X, y=None):
        # No aprende nada — transformación determinista
        return self

    def transform(self, X, y=None):
        X = X.copy()                                       # No modificar el DataFrame original

        # Construcción de CLTV con control de valores problemáticos
        X["CLTV"] = np.where(
            (X["VALUE"].notna()) & (X["MORTDUE"].notna()) & (X["VALUE"] > 0),
            (X["LOAN"] + X["MORTDUE"]) / X["VALUE"],
            np.nan,                                        # Si faltan datos o VALUE=0, NaN
        )

        # Cap económico
        X["CLTV"] = X["CLTV"].clip(upper=self.cap)

        # MORTDUE y VALUE ya están capturadas por CLTV
        X = X.drop(columns=["MORTDUE", "VALUE"])

        return X

    def get_feature_names_out(self, input_features=None):
        """Permite a ColumnTransformer recuperar los nombres de columnas post-transformación."""
        if input_features is None:
            raise ValueError("input_features es necesario — pasa los nombres de columnas de entrada")
        features = [f for f in input_features if f not in ["MORTDUE", "VALUE"]]
        features = list(features) + ["CLTV"]
        return np.array(features, dtype=object)