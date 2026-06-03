"""
ÉTAPE 2 — Estimation du VECM
Entrée  : data_gazole_brent.csv
Sorties : resultats_vecm.txt  (résumé complet lisible)
          previsions_vecm.csv (prévisions 12 mois)
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from vecm import VECM, select_order, select_coint_rank

# =============================================================================
# 1. CHARGEMENT
# =============================================================================
df = pd.read_csv('data_gazole_brent.csv', index_col=0, parse_dates=True)
df.index = pd.PeriodIndex(df.index, freq='M')

# =============================================================================
# 2. SÉLECTION DES HYPERPARAMÈTRES
# =============================================================================
lag_results  = select_order(df, maxlags=6, deterministic="co")
k            = lag_results.aic

rank_result  = select_coint_rank(df, det_order=1, k_ar_diff=k, method="trace")
r            = rank_result.rank if rank_result.rank > 0 else 1

# =============================================================================
# 3. ESTIMATION
# =============================================================================
model  = VECM(endog=df, k_ar_diff=k, coint_rank=r, deterministic="coli")
result = model.fit()

# =============================================================================
# 4. PRÉVISIONS 12 MOIS
# =============================================================================
forecast     = result.predict(steps=12)
last_period  = df.index[-1]
future_index = pd.period_range(start=last_period + 1, periods=12, freq='M')
prev_df      = pd.DataFrame(forecast, index=future_index, columns=df.columns)
prev_df.to_csv('previsions_vecm.csv')

# =============================================================================
# 5. RÉDACTION DU FICHIER DE RÉSULTATS LISIBLE
# =============================================================================
lines = []

def h(title):
    lines.append("")
    lines.append("=" * 65)
    lines.append(f"  {title}")
    lines.append("=" * 65)

def sub(title):
    lines.append("")
    lines.append(f"--- {title} ---")

# ── En-tête ────────────────────────────────────────────────────────
lines.append("RÉSULTATS DU VECM — Prix du gazole et du Brent en France")
lines.append(f"Période : {df.index[0]} → {df.index[-1]}  ({len(df)} observations mensuelles)")
lines.append(f"Modèle  : VECM(k={k})  |  Rang de cointégration r={r}")

# ── 1. Paramètres du modèle ─────────────────────────────────────────
h("1. PARAMÈTRES DU MODÈLE")

lines.append(f"  Nombre de lags retenus (AIC) : {k}")
lines.append(f"  Rang de cointégration        : {r}")
if rank_result.rank == 0:
    lines.append("  ⚠ Le test de Johansen suggère r=0 (pas de cointégration")
    lines.append("    statistiquement détectée). r=1 est forcé sur la base")
    lines.append("    de l'hypothèse économique : le Brent détermine le gazole")
    lines.append("    à long terme (ruptures 2008/2020/2022 peuvent biaiser le test).")

# ── 2. Relation de long terme ────────────────────────────────────────
h("2. RELATION D'ÉQUILIBRE DE LONG TERME (vecteur bêta)")

beta  = result.beta
alpha = result.alpha

lines.append("")
lines.append("  Équation normalisée (coefficient gazole = 1) :")
lines.append(f"    gazole_ttc = {-beta[1,0]:.4f} × brent_eur_l + constante de tendance")
lines.append("")
lines.append("  Interprétation :")
coef_brent = -beta[1, 0]
if coef_brent > 0:
    lines.append(f"    → Une hausse de 0.10 €/L du Brent est associée à une hausse")
    lines.append(f"      de {coef_brent * 0.10:.4f} €/L du gazole à long terme.")
lines.append(f"    → Ce coefficient reflète la transmission structurelle du coût")
lines.append(f"      du brut dans le prix final TTC (raffinage + distribution + taxes).")

# ── 3. Vitesse d'ajustement ──────────────────────────────────────────
h("3. VITESSE D'AJUSTEMENT (coefficient alpha)")

alpha_gazole = float(result.alpha[0, 0])
alpha_brent  = float(result.alpha[1, 0])

lines.append(f"  alpha (gazole) = {alpha_gazole:.4f}")
lines.append(f"  alpha (Brent)  = {alpha_brent:.4f}")
lines.append("")

if alpha_gazole < 0:
    mois_retour = abs(round(1 / alpha_gazole))
    lines.append(f"  ✓ Le signe négatif d'alpha (gazole) confirme qu'il existe")
    lines.append(f"    une force de rappel vers l'équilibre.")
    lines.append(f"  → Le gazole corrige {abs(alpha_gazole)*100:.1f}% de son écart chaque mois.")
    lines.append(f"  → Un choc met environ {mois_retour} mois à se résorber à moitié.")
else:
    lines.append("  ⚠ Alpha positif : pas de force de rappel détectée sur le gazole.")

if alpha_brent < 0:
    lines.append(f"  → Le Brent s'ajuste aussi ({abs(alpha_brent)*100:.1f}%/mois) : la causalité")
    lines.append(f"    est bidirectionnelle, mais le gazole suit davantage le Brent.")

# ── 4. Dynamique de court terme ──────────────────────────────────────
h("4. DYNAMIQUE DE COURT TERME (coefficients Gamma)")

sub("Équation du gazole")
gamma = result.gamma  # shape (neqs, neqs*(k-1)) si k>1, sinon vide
det   = result.det_coef

lines.append(f"  const            = {float(result.const[0]):.4f}")
if result.gamma.size > 0:
    lines.append(f"  L1.gazole_ttc    = {result.gamma[0, 0]:.4f}")
    lines.append(f"  L1.brent_eur_l   = {result.gamma[0, 1]:.4f}")
    lines.append("")
    coef_g = result.gamma[0, 1]
    lines.append(f"  → Une hausse de 0.10 €/L du Brent le mois précédent")
    lines.append(f"    entraîne une variation de {coef_g * 0.10:+.4f} €/L du gazole le mois suivant.")
else:
    lines.append("  (k=1 : pas de termes Gamma — toute la dynamique passe par alpha×beta)")

# ── 5. Prévisions ────────────────────────────────────────────────────
h("5. PRÉVISIONS SUR 12 MOIS (à partir de " + str(last_period + 1) + ")")

lines.append("")
lines.append(f"  {'Mois':<12} {'Gazole TTC (€/L)':>18} {'Brent (€/L)':>14}")
lines.append(f"  {'-'*12} {'-'*18} {'-'*14}")
for period, row in prev_df.iterrows():
    lines.append(f"  {str(period):<12} {row['gazole_ttc']:>18.4f} {row['brent_eur_l']:>14.4f}")

lines.append("")
lines.append(f"  Gazole en {str(future_index[-1])} : {prev_df['gazole_ttc'].iloc[-1]:.4f} €/L")
delta = prev_df['gazole_ttc'].iloc[-1] - df['gazole_ttc'].iloc[-1]
lines.append(f"  Variation prévue vs avril 2026 : {delta:+.4f} €/L ({delta/df['gazole_ttc'].iloc[-1]*100:+.1f}%)")

# ── 6. Mise en garde ─────────────────────────────────────────────────
h("6. LIMITES ET MISE EN GARDE")
lines.append("")
lines.append("  • Ces prévisions supposent que le Brent reste proche de son")
lines.append("    niveau actuel. Tout choc géopolitique (Moyen-Orient, Russie)")
lines.append("    ou macroéconomique (récession, dollar fort) les invaliderait.")
lines.append("")
lines.append("  • La fiscalité (TICPE) n'est pas modélisée ici. Une réforme")
lines.append("    fiscale constituerait un levier immédiat sur le prix final.")
lines.append("")
lines.append("  • Le VECM ne capture pas les asymétries : la littérature montre")
lines.append("    que les hausses du Brent se transmettent plus vite que les baisses")
lines.append("    ('rocket and feather effect', Bacon 1991).")

# ── Écriture du fichier ──────────────────────────────────────────────
output = "\n".join(lines)
with open("resultats_vecm.txt", "w", encoding="utf-8") as f:
    f.write(output)

print(output)
print("\n✓ Fichier créé : resultats_vecm.txt")
print("✓ Fichier créé : previsions_vecm.csv")
