"""
ÉTAPE 1 — Préparation des données
Produit : data_gazole_brent.csv (2 colonnes : gazole_ttc, brent_eur_l)
"""
import pandas as pd

# --- Prix du gazole (hebdo → mensuel) ---
diesel = pd.read_csv('prix_diesel_france.csv', parse_dates=['date'])
diesel = diesel.set_index('date').resample('MS').mean()
diesel.index.name = 'date'

# --- Brent en $/L ---
brent = pd.read_csv('brent_usd_litre.csv', parse_dates=['Date'])
brent = brent.rename(columns={'Date': 'date', 'Brent_USD_litre': 'brent_usd_l'})
brent['date'] = brent['date'].values.astype('datetime64[M]').astype('datetime64[ns]')
brent = brent.set_index('date')

# --- EUR/USD pour convertir le Brent en euros ---
eurusd = pd.read_csv('eur_usd_bce.csv', parse_dates=['Date'])
eurusd = eurusd.rename(columns={'Date': 'date', 'EUR_USD': 'eur_usd'})
eurusd['date'] = eurusd['date'].values.astype('datetime64[M]').astype('datetime64[ns]')
eurusd = eurusd.set_index('date')

# --- Fusion et conversion Brent en €/L ---
df = diesel.join(brent).join(eurusd)
df['brent_eur_l'] = df['brent_usd_l'] / df['eur_usd']

# --- Dataset final : uniquement gazole et Brent en euros ---
df_final = df[['prix_diesel_france_eur_L', 'brent_eur_l']].dropna()
df_final.columns = ['gazole_ttc', 'brent_eur_l']

df_final.to_csv('data_gazole_brent.csv')

print("✓ Fichier créé : data_gazole_brent.csv")
print(f"  Période  : {df_final.index[0].strftime('%Y-%m')} → {df_final.index[-1].strftime('%Y-%m')}")
print(f"  Obs.     : {len(df_final)} mois")
print(f"  NaN      : {df_final.isna().sum().to_dict()}")
print()
print(df_final.head(5).round(4))
