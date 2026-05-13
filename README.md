# tfm-data-science-credit-risk-karim
Modelo de scoring crediticio (PD): Logistic Regression + WoE frente benchmark LightGBM, XGBoost.  Evaluación con AUC, KS, Gini, PSI y SHAP.

## Known Issues

### Warnings suprimidos en notebooks

En `05_Comparacion_Modelos_PD.ipynb` se suprimen dos warnings de forma controlada. Ambos se han validado como inocuos, pero quedan pendientes de resolver antes de una eventual puesta en producción:

**1. `FutureWarning: This Pipeline instance is not fitted yet`**

- **Origen**: scikit-learn ≥ 1.6
- **Causa**: al deserializar un `Pipeline` con transformers personalizados (`CLTVTransformer`), el atributo interno `_is_fitted` no siempre se reconstruye correctamente desde pickle.
- **Impacto actual**: ninguno — el pipeline predice correctamente. Verificado contra los datos de entrenamiento.
- **Impacto futuro**: en scikit-learn 1.8 este warning pasará a ser un error. Habrá que implementar un `_is_fitted = True` manual en `CLTVTransformer` o migrar a `__sklearn_is_fitted__`.
- **Referencia**: [sklearn deprecation cycle](https://scikit-learn.org/stable/developers/develop.html#pandas_out)

**2. `UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names`**

- **Origen**: `lightgbm` al pasar datos sin nombres de columna a un modelo entrenado con DataFrame.
- **Causa**: el `ColumnTransformer` dentro del pipeline devuelve un `numpy.ndarray` sin nombres de columna. LightGBM los infirió durante el entrenamiento pero no puede validarlos en inferencia.
- **Impacto**: ninguno — el orden posicional de columnas se preserva.
- **Solución futura**: configurar `ColumnTransformer(..., verbose_feature_names_out=False)` o usar `set_output(transform="pandas")`.

Estos warnings se suprimen explícitamente en una celda dedicada al inicio del notebook 05, **por mensaje concreto** (nunca de forma genérica), para preservar la visibilidad de cualquier otro warning legítimo.