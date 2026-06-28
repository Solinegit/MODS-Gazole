import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Chargement du prix diesel FR TTC depuis le fichier gazole.xlsx
df_raw = pd.read_excel('Données/gazole.xlsx', sheet_name='Prices with taxes', header=None)

# Localisation de la colonne FR diesel TTC (en €/1000L)
row0 = df_raw.iloc[0]
fr_diesel_col = next(i for i, v in enumerate(row0) if 'FR_price_with_tax_diesel' in str(v))

# Extraction des données (à partir de la ligne 3, après les 3 lignes de headers)
df_diesel = df_raw.iloc[3:].rename(columns={0: 'date', fr_diesel_col: 'prix_diesel_france_eur_L'})
df_diesel['date'] = pd.to_datetime(df_diesel['date'], errors='coerce')
df_diesel['prix_diesel_france_eur_L'] = pd.to_numeric(df_diesel['prix_diesel_france_eur_L'], errors='coerce') / 1000
df_diesel = df_diesel[['date', 'prix_diesel_france_eur_L']].dropna().sort_values('date')

# 2. Chargement du cours du Brent (EUR/baril → USD/litre)
df_brent = pd.read_csv('data/DCOILBRENTEU.csv')
df_brent['date'] = pd.to_datetime(df_brent['observation_date'])
# 1 baril = 158.987 litres, taux EUR/USD moyen 2022 ≈ 1.05
df_brent['Brent_USD_litre'] = df_brent['BRENT'] / 158.987 * 1.05
df_brent = df_brent[['date', 'Brent_USD_litre']].sort_values('date')

# 3. Fusion des deux séries
df = pd.merge_asof(df_diesel, df_brent, on='date', direction='backward')

# Isolation de l'année 2022
df_2022 = df[df['date'].dt.year == 2022].copy()

# 4. Paramètres de la simulation
volume_hebdo = 42.5e9 / 52  # 42,5 milliards de litres annuels
TICPE_FIXE = 0.608
prix_cible = 1.60            # Prix TTC de référence à la pompe (€/L)

# 5. Calcul de la TICPE flottante
# Quand le prix TTC dépasse la cible, la TICPE baisse proportionnellement
df_2022['ticpe_flottante'] = TICPE_FIXE - ((df_2022['prix_diesel_france_eur_L'] - prix_cible) / 1.20)
df_2022['ticpe_flottante'] = df_2022['ticpe_flottante'].clip(lower=0.33, upper=TICPE_FIXE)

# 6. Calcul des revenus cumulés (Fixe vs Flottant)
revenu_fixe_hebdo = pd.Series([TICPE_FIXE * volume_hebdo] * len(df_2022), index=df_2022.index)
df_2022['cumul_revenu_fixe'] = revenu_fixe_hebdo.cumsum() / 1e9
df_2022['cumul_revenu_flottant'] = (df_2022['ticpe_flottante'] * volume_hebdo).cumsum() / 1e9

# 7. Figure avec 2 graphiques
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

# --- GRAPHIQUE DU HAUT : Recettes cumulées et Brent ---
ax1.fill_between(df_2022['date'], df_2022['cumul_revenu_fixe'], df_2022['cumul_revenu_flottant'],
                 color='red', alpha=0.15, label='Manque à gagner budgétaire')
ax1.plot(df_2022['date'], df_2022['cumul_revenu_fixe'], label='Recettes État (TICPE Fixe)', color='blue', linewidth=2)
ax1.plot(df_2022['date'], df_2022['cumul_revenu_flottant'], label='Recettes État (TICPE Flottante)', color='red', linestyle='--', linewidth=2)
ax1.set_ylabel('Recettes cumulées (Milliards €)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(df_2022['date'], df_2022['Brent_USD_litre'], label='Cours du Brent ($/Litre)', color='green', linewidth=2, linestyle='-.')
ax2.set_ylabel('Prix du Brent ($/Litre)', fontsize=12, color='green')
ax2.tick_params(axis='y', labelcolor='green')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
ax1.set_title("Impact du cours du Brent sur le déficit budgétaire d'une TICPE flottante (2022)", fontsize=14, fontweight='bold')

# --- GRAPHIQUE DU BAS : Évolution de la TICPE ---
ax3.plot(df_2022['date'], [TICPE_FIXE]*len(df_2022), label='TICPE Fixe Légale (0.608 €/L)', color='blue', linewidth=2)
ax3.plot(df_2022['date'], df_2022['ticpe_flottante'], label='TICPE Flottante Appliquée (€/L)', color='purple', linewidth=2)
ax3.fill_between(df_2022['date'], [TICPE_FIXE]*len(df_2022), df_2022['ticpe_flottante'],
                 color='purple', alpha=0.15, label='Baisse de taxe (Cadeau fiscal)')

ax3.set_ylabel('Montant de la TICPE (€/Litre)', fontsize=12)
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylim(0.30, 0.65)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='lower left')

plt.tight_layout()
plt.savefig('data/ticpe_output_v2.png', dpi=150, bbox_inches='tight')
plt.show()