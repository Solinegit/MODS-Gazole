import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================
diesel = pd.read_csv('prix_diesel_france.csv', parse_dates=['date'])
diesel = diesel.set_index('date').resample('MS').mean()

brent = pd.read_csv('brent_usd_litre.csv', parse_dates=['Date'])
brent = brent.rename(columns={'Date': 'date', 'Brent_USD_litre': 'brent_usd_l'})
brent['date'] = brent['date'].values.astype('datetime64[M]').astype('datetime64[ns]')
brent = brent.set_index('date')

eurusd = pd.read_csv('eur_usd_bce.csv', parse_dates=['Date'])
eurusd = eurusd.rename(columns={'Date': 'date', 'EUR_USD': 'eur_usd'})
eurusd['date'] = eurusd['date'].values.astype('datetime64[M]').astype('datetime64[ns]')
eurusd = eurusd.set_index('date')

df = diesel.join(brent).join(eurusd)
df['brent_eur_l'] = df['brent_usd_l'] / df['eur_usd']
df = df[['prix_diesel_france_eur_L', 'brent_eur_l', 'eur_usd']].dropna()
df.columns = ['gazole_ttc', 'brent_eur_l', 'eur_usd']

# Événements clés à annoter
events = {
    '2008-07': 'Pic Brent\n(147 $/baril)',
    '2009-01': 'Crise\nfinancière',
    '2020-04': 'COVID-19',
    '2022-03': 'Guerre\nUkraine',
}

# =============================================================================
# FIGURE : 3 SOUS-GRAPHIQUES EMPILÉS + AXES SECONDAIRES
# =============================================================================
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 1, figure=fig, hspace=0.45)

colors = {
    'gazole': '#2166ac',
    'brent':  '#d6604d',
    'eurusd': '#4dac26',
    'event':  '#888888',
}

# ── Graphique 1 : Prix du gazole TTC ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['gazole_ttc'], color=colors['gazole'], linewidth=1.8, label='Gazole TTC (€/L)')
ax1.set_ylabel('€ / litre', fontsize=10)
ax1.set_title('Prix du gazole TTC en France (2007–2026)', fontsize=11, fontweight='bold')
ax1.set_ylim(0.8, 2.5)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.tick_params(labelbottom=False)

for date_str, label in events.items():
    ts = pd.Timestamp(date_str)
    if ts in df.index:
        ax1.axvline(ts, color=colors['event'], linewidth=0.8, linestyle='--', alpha=0.7)
        ax1.text(ts, 2.42, label, fontsize=7, color=colors['event'],
                 ha='center', va='top', rotation=0,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

# ── Graphique 2 : Brent en €/L ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(df.index, df['brent_eur_l'], color=colors['brent'], linewidth=1.8, label='Brent (€/L)')
ax2.fill_between(df.index, df['brent_eur_l'], alpha=0.15, color=colors['brent'])
ax2.set_ylabel('€ / litre', fontsize=10)
ax2.set_title('Cours du Brent converti en €/L', fontsize=11, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.tick_params(labelbottom=False)

for date_str in events:
    ts = pd.Timestamp(date_str)
    if ts in df.index:
        ax2.axvline(ts, color=colors['event'], linewidth=0.8, linestyle='--', alpha=0.7)

# ── Graphique 3 : Taux EUR/USD ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.plot(df.index, df['eur_usd'], color=colors['eurusd'], linewidth=1.8, label='EUR/USD')
ax3.axhline(1.0, color='black', linewidth=0.7, linestyle=':', alpha=0.5)
ax3.text(df.index[2], 1.01, 'parité', fontsize=8, color='black', alpha=0.6)
ax3.set_ylabel('USD par EUR', fontsize=10)
ax3.set_title('Taux de change EUR/USD (BCE)', fontsize=11, fontweight='bold')
ax3.set_ylim(0.85, 1.65)
ax3.grid(axis='y', linestyle='--', alpha=0.4)

for date_str in events:
    ts = pd.Timestamp(date_str)
    if ts in df.index:
        ax3.axvline(ts, color=colors['event'], linewidth=0.8, linestyle='--', alpha=0.7)

# Format des dates sur l'axe X (uniquement ax3 visible)
import matplotlib.dates as mdates
ax3.xaxis.set_major_locator(mdates.YearLocator(2))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax3.get_xticklabels(), rotation=0, ha='center')

# Légende commune en bas
legend_patches = [
    mpatches.Patch(color=colors['gazole'], label='Gazole TTC (€/L)'),
    mpatches.Patch(color=colors['brent'],  label='Brent (€/L)'),
    mpatches.Patch(color=colors['eurusd'], label='EUR/USD'),
    mpatches.Patch(color=colors['event'],  label='Événement majeur', linestyle='--'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=4,
           fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

fig.suptitle('Déterminants du prix du gazole en France — VECM 2007–2026',
             fontsize=13, fontweight='bold', y=1.01)

plt.savefig('graphique_vecm.png', dpi=150, bbox_inches='tight')
print("✓ Graphique sauvegardé : graphique_vecm.png")
plt.show()