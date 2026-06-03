"""
Pipeline de préparation des données pour le VECM
Prix du gazole en France, 2007-2026
"""

import pandas as pd
import numpy as np

# =============================================================================
# 1. CHARGEMENT ET NETTOYAGE DES SOURCES
# =============================================================================

# --- Prix du gazole en France (hebdomadaire → mensuel) ---
diesel = pd.read_csv('prix_diesel_france.csv', parse_dates=['date'])
diesel = diesel.set_index('date').resample('MS').mean()  # moyenne mensuelle
diesel.index.name = 'date'

# --- Cours du Brent (mensuel, en $/L) ---
brent = pd.read_csv('brent_usd_litre.csv', parse_dates=['Date'])
brent = brent.rename(columns={'Date': 'date', 'Brent_USD_litre': 'brent_usd_l'})
brent['date'] = brent['date'].values.astype('datetime64[M]').astype('datetime64[ns]')
brent = brent.set_index('date')

# --- Taux de change EUR/USD (mensuel, source BCE) ---
eurusd = pd.read_csv('eur_usd_bce.csv', parse_dates=['Date'])
eurusd = eurusd.rename(columns={'Date': 'date', 'EUR_USD': 'eur_usd'})
eurusd['date'] = eurusd['date'].values.astype('datetime64[M]').astype('datetime64[ns]')
eurusd = eurusd.set_index('date')

# =============================================================================
# 2. FUSION ET CALCUL DU BRENT EN EUROS
# =============================================================================

df = diesel.join(brent).join(eurusd)

# Conversion Brent $/L → €/L (le pétrole se cote en $, on ramène en €)
df['brent_eur_l'] = df['brent_usd_l'] / df['eur_usd']

# =============================================================================
# 3. DATAFRAME FINAL POUR LE VECM
# =============================================================================

# Colonnes retenues pour le VECM :
#   - gazole_ttc  : prix du gazole TTC en France (€/L)
#   - brent_eur_l : cours du Brent converti en €/L
#   - eur_usd     : taux de change EUR/USD
#
# Note : la TICPE est quasi-constante par paliers → à passer en exog si besoin

df_vecm = df[['prix_diesel_france_eur_L', 'brent_eur_l', 'eur_usd']].copy()
df_vecm.columns = ['gazole_ttc', 'brent_eur_l', 'eur_usd']
df_vecm = df_vecm.dropna()

# Index en PeriodIndex mensuel (pratique pour statsmodels)
df_vecm.index = pd.PeriodIndex(df_vecm.index, freq='M')

print("=== Aperçu du DataFrame ===")
print(df_vecm.head(5))
print("...")
print(df_vecm.tail(3))
print(f"\nShape     : {df_vecm.shape}  ({df_vecm.shape[0]} mois)")
print(f"Période   : {df_vecm.index[0]} → {df_vecm.index[-1]}")
print(f"NaN       : {df_vecm.isna().sum().to_dict()}")
print("\nStatistiques descriptives :")
print(df_vecm.describe().round(4))

# =============================================================================
# 4. EXPORT CSV PROPRE (pour réutilisation)
# =============================================================================

df_vecm.to_csv('data_vecm_gazole.csv')
print("\n✓ Fichier exporté : data_vecm_gazole.csv")

# =============================================================================
# 5. UTILISATION AVEC LE VECM
# =============================================================================

# from vecm import VECM, select_order, select_coint_rank
#
# # Choisir le nombre de lags
# lag_results = select_order(df_vecm, maxlags=6, deterministic="co")
# k = lag_results.aic  # typiquement 1 ou 2
#
# # Rang de cointégration (test de Johansen)
# rank_result = select_coint_rank(df_vecm, det_order=0, k_ar_diff=k)
# r = rank_result.rank
#
# # Estimation
# model = VECM(endog=df_vecm, k_ar_diff=k, coint_rank=r, deterministic="co")
# result = model.fit()
# print(result.summary())
#
# # Prévision sur 12 mois
# forecast = result.predict(steps=12)