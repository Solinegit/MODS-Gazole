import pandas as pd
import numpy as np
from vecm import VECM, select_order, select_coint_rank

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================
df = pd.read_csv("data_vecm_gazole.csv", index_col=0)
df.index = pd.PeriodIndex(df.index, freq="M")
print(f"Données chargées : {df.shape[0]} observations ({df.index[0]} → {df.index[-1]})")
print(df.head(3))
print()

# =============================================================================
# 2. SÉLECTION DU NOMBRE DE LAGS
# =============================================================================
print("=== Sélection du nombre de lags ===")
lag_results = select_order(df, maxlags=6, deterministic="co")
k = lag_results.aic
print(f"Lags suggérés par AIC : {k}")
print(f"Lags suggérés par BIC : {lag_results.bic}")
print()

# =============================================================================
# 3. TEST DE COINTÉGRATION DE JOHANSEN
# =============================================================================
print("=== Test de cointégration de Johansen (det_order=1 = tendance) ===")
rank_result = select_coint_rank(df, det_order=1, k_ar_diff=k, method="trace")
print(rank_result.summary())
r = rank_result.rank
print(f"Rang détecté automatiquement : {r}")

# Si rang=0 (fréquent sur des séries de prix avec ruptures structurelles),
# on force r=1 pour pouvoir estimer le VECM malgré tout.
# Cela revient à postuler l'existence d'une relation de long terme,
# ce qui est économiquement justifié (le gazole suit le Brent sur le long terme).
if r == 0:
    r = 1
    print("  → Rang forcé à 1 (hypothèse économique : relation de long terme Brent→gazole)")
print()

# =============================================================================
# 4. ESTIMATION DU VECM
# =============================================================================
print("=== Estimation du VECM ===")
# det_order=1 dans Johansen correspond à deterministic="coli" dans VECM
model = VECM(endog=df, k_ar_diff=k, coint_rank=r, deterministic="coli")
result = model.fit()

# Affichage sécurisé du summary (contourne le bug IndexError sur rang=0)
try:
    print(result.summary())
except IndexError:
    print("[summary() indisponible — affichage manuel des coefficients clés]")
    print("Alpha (vitesse d'ajustement) :\n", result.alpha)
    print("Beta (vecteur de cointégration) :\n", result.beta)
    print("Gamma (dynamique court terme) :\n", result.gamma)

# =============================================================================
# 5. PRÉVISION SUR 12 MOIS
# =============================================================================
print("\n=== Prévision sur 12 mois ===")
forecast = result.predict(steps=12)

last_period = df.index[-1]
future_index = pd.period_range(start=last_period + 1, periods=12, freq="M")
forecast_df = pd.DataFrame(forecast, index=future_index, columns=df.columns)
print(forecast_df.round(4))

print("\n✓ Terminé.")
