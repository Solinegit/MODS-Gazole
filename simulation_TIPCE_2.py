import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Chargement du prix diesel FR TTC depuis gazole.xlsx
df_raw = pd.read_excel('Données/gazole.xlsx', sheet_name='Prices with taxes', header=None)
row0 = df_raw.iloc[0]
fr_diesel_col = next(i for i, v in enumerate(row0) if 'FR_price_with_tax_diesel' in str(v))

df_diesel = df_raw.iloc[3:].rename(columns={0: 'date', fr_diesel_col: 'prix_diesel_france_eur_L'})
df_diesel['date'] = pd.to_datetime(df_diesel['date'], errors='coerce')
df_diesel['prix_diesel_france_eur_L'] = pd.to_numeric(df_diesel['prix_diesel_france_eur_L'], errors='coerce') / 1000
df_diesel = df_diesel[['date', 'prix_diesel_france_eur_L']].dropna().sort_values('date')

# 2. Chargement du Brent (EUR/baril → USD/litre)
df_brent = pd.read_csv('data/DCOILBRENTEU.csv')
df_brent['date'] = pd.to_datetime(df_brent['observation_date'])
df_brent['Brent_USD_litre'] = df_brent['BRENT'] / 158.987 * 1.05
df_brent = df_brent[['date', 'Brent_USD_litre']].sort_values('date')

# 3. Fusion
df = pd.merge_asof(df_diesel, df_brent, on='date', direction='backward')
df_period = df[df['date'].dt.year.isin([2023, 2024, 2025])].copy()

# 4. Paramètres
volume_hebdo = 42.5e9 / 52
TICPE_FIXE = 0.608
prix_cible_2023 = 1.767  # 1er relevé janvier 2023
prix_cible_2022 = 1.60   # Référence du graphe 2022

# 5. TICPE flottante avec référence 2023 (plancher uniquement, pas de plafond)
df_period['ticpe_flottante_2023'] = TICPE_FIXE - ((df_period['prix_diesel_france_eur_L'] - prix_cible_2023) / 1.20)
df_period['ticpe_flottante_2023'] = df_period['ticpe_flottante_2023'].clip(lower=0.33)

# 6. TICPE flottante avec référence 2022 à 1.60€ (plancher uniquement)
df_period['ticpe_flottante_1_60'] = TICPE_FIXE - ((df_period['prix_diesel_france_eur_L'] - prix_cible_2022) / 1.20)
df_period['ticpe_flottante_1_60'] = df_period['ticpe_flottante_1_60'].clip(lower=0.33)

# 7. Recettes cumulées
revenu_fixe_hebdo = pd.Series([TICPE_FIXE * volume_hebdo] * len(df_period), index=df_period.index)
df_period['cumul_revenu_fixe']       = revenu_fixe_hebdo.cumsum() / 1e9
df_period['cumul_revenu_flottant_2023'] = (df_period['ticpe_flottante_2023'] * volume_hebdo).cumsum() / 1e9
df_period['cumul_revenu_flottant_1_60'] = (df_period['ticpe_flottante_1_60'] * volume_hebdo).cumsum() / 1e9

# 8. Figure
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

# --- GRAPHIQUE DU HAUT : Recettes cumulées ---
# Zones ref 2023
ax1.fill_between(df_period['date'], df_period['cumul_revenu_fixe'], df_period['cumul_revenu_flottant_2023'],
                 where=df_period['cumul_revenu_flottant_2023'] >= df_period['cumul_revenu_fixe'],
                 color='green', alpha=0.12, label='Surplus (réf. 1.767€)')
ax1.fill_between(df_period['date'], df_period['cumul_revenu_fixe'], df_period['cumul_revenu_flottant_2023'],
                 where=df_period['cumul_revenu_flottant_2023'] < df_period['cumul_revenu_fixe'],
                 color='red', alpha=0.12, label='Manque à gagner (réf. 1.767€)')

ax1.plot(df_period['date'], df_period['cumul_revenu_fixe'],
         label='Recettes État (TICPE Fixe)', color='blue', linewidth=2)
ax1.plot(df_period['date'], df_period['cumul_revenu_flottant_2023'],
         label='TICPE Flottante — réf. 1.767 €/L (jan. 2023)', color='red', linestyle='--', linewidth=2)
ax1.plot(df_period['date'], df_period['cumul_revenu_flottant_1_60'],
         label='TICPE Flottante — réf. 1.60 €/L (comme 2022)', color='orange', linestyle=':', linewidth=2.5)

ax1.set_ylabel('Recettes cumulées (Milliards €)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(df_period['date'], df_period['Brent_USD_litre'],
         label='Cours du Brent ($/Litre)', color='green', linewidth=2, linestyle='-.')
ax2.set_ylabel('Prix du Brent ($/Litre)', fontsize=12, color='green')
ax2.tick_params(axis='y', labelcolor='green')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=9)
ax1.set_title("Bilan budgétaire d'une TICPE flottante (2023-2025)\n"
              "Comparaison : référence 1.767 €/L (jan. 2023) vs référence 1.60 €/L (comme 2022)",
              fontsize=13, fontweight='bold')

# --- GRAPHIQUE DU BAS : Niveaux de TICPE ---
ax3.plot(df_period['date'], [TICPE_FIXE]*len(df_period),
         label=f'TICPE Fixe Légale ({TICPE_FIXE} €/L)', color='blue', linewidth=2)
ax3.plot(df_period['date'], df_period['ticpe_flottante_2023'],
         label='TICPE Flottante — réf. 1.767 €/L', color='purple', linewidth=2)
ax3.plot(df_period['date'], df_period['ticpe_flottante_1_60'],
         label='TICPE Flottante — réf. 1.60 €/L', color='orange', linewidth=2, linestyle=':')

ax3.fill_between(df_period['date'], TICPE_FIXE, df_period['ticpe_flottante_2023'],
                 where=df_period['ticpe_flottante_2023'] < TICPE_FIXE,
                 color='red', alpha=0.10)
ax3.fill_between(df_period['date'], TICPE_FIXE, df_period['ticpe_flottante_2023'],
                 where=df_period['ticpe_flottante_2023'] >= TICPE_FIXE,
                 color='green', alpha=0.10)

ax3.set_ylabel('Montant de la TICPE (€/Litre)', fontsize=12)
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylim(0.30, 0.88)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right', fontsize=9)

# Séparateurs d'années
for year in [2023, 2024, 2025]:
    ax1.axvline(pd.Timestamp(f'{year}-01-01'), color='gray', linestyle=':', alpha=0.6)
    ax3.axvline(pd.Timestamp(f'{year}-01-01'), color='gray', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('data/ticpe_output_2023_2025.png', dpi=150, bbox_inches='tight')
plt.show()