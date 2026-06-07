"""
ÉTAPE 3 — Validation du VECM
3 tests essentiels : normalité, blancheur des résidus, stabilité
Entrée : data_gazole_brent.csv + vecm.py
Sortie : validation_vecm.txt + graphique_validation.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import sys
sys.path.insert(0, '.')
from vecm import VECM

# =============================================================================
# 1. ESTIMATION
# =============================================================================
df = pd.read_csv('data_gazole_brent.csv', index_col=0, parse_dates=True)
df.index = pd.PeriodIndex(df.index, freq='M')

model  = VECM(endog=df, k_ar_diff=1, coint_rank=1, deterministic="coli")
result = model.fit()

resid         = result.resid          # (nobs, 2)
resid_gazole  = resid[:, 0]
resid_brent   = resid[:, 1]
nobs          = len(resid_gazole)

# =============================================================================
# 2. TEST 1 — NORMALITÉ DES RÉSIDUS (Jarque-Bera)
# =============================================================================
jb_g, pjb_g = stats.jarque_bera(resid_gazole)
jb_b, pjb_b = stats.jarque_bera(resid_brent)
skew_g       = stats.skew(resid_gazole)
kurt_g       = stats.kurtosis(resid_gazole)
skew_b       = stats.skew(resid_brent)
kurt_b       = stats.kurtosis(resid_brent)

# =============================================================================
# 3. TEST 2 — BLANCHEUR DES RÉSIDUS (Ljung-Box sur 12 lags)
# =============================================================================
from statsmodels.tsa.stattools import acf as sm_acf
lb_g  = sm_acf(resid_gazole, nlags=12, fft=True)
lb_b  = sm_acf(resid_brent,  nlags=12, fft=True)

# Ljung-Box Q-stat manual
def ljung_box_q(resid, nlags=12):
    n   = len(resid)
    acf_vals = [np.corrcoef(resid[k:], resid[:-k])[0,1] for k in range(1, nlags+1)]
    Q   = n * (n + 2) * sum(r**2 / (n - k) for k, r in enumerate(acf_vals, 1))
    pval = 1 - stats.chi2.cdf(Q, df=nlags)
    return Q, pval, acf_vals

Q_g, pQ_g, acf_g = ljung_box_q(resid_gazole)
Q_b, pQ_b, acf_b = ljung_box_q(resid_brent)

# =============================================================================
# 4. TEST 3 — STABILITÉ (racines du VAR associé)
# =============================================================================
var_rep   = result.var_rep   # (k_ar, neqs, neqs)
k_ar, K   = var_rep.shape[0], var_rep.shape[1]
companion = np.zeros((K * k_ar, K * k_ar))
for i in range(k_ar):
    companion[:K, i*K:(i+1)*K] = var_rep[i]
if k_ar > 1:
    companion[K:, :K*(k_ar-1)] = np.eye(K * (k_ar - 1))

eigenvalues = np.linalg.eigvals(companion)
moduli      = np.abs(eigenvalues)
stable      = np.all(moduli <= 1.0 + 1e-10)

# =============================================================================
# 5. GRAPHIQUE (2×2)
# =============================================================================
fig = plt.figure(figsize=(13, 10))
fig.suptitle("Validation du VECM — 3 tests essentiels", fontsize=13, fontweight='bold')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

C1, C2 = '#2166ac', '#d6604d'

# ── A. QQ-plot résidus gazole ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
(osm, osr), (slope, intercept, r) = stats.probplot(resid_gazole)
ax1.plot(osm, osr,           'o', color=C1, markersize=3, alpha=0.6, label='Résidus gazole')
ax1.plot(osm, slope*np.array(osm)+intercept, '--', color='black', linewidth=1.2)
ax1.set_title('Test normalité — QQ-plot (gazole)', fontsize=10, fontweight='bold')
ax1.set_xlabel('Quantiles théoriques (loi normale)')
ax1.set_ylabel('Quantiles observés')
verdict_jb_g = "✓ Normale" if pjb_g > 0.05 else "✗ Non normale"
ax1.text(0.05, 0.92, f'Jarque-Bera p={pjb_g:.4f}  {verdict_jb_g}',
         transform=ax1.transAxes, fontsize=8.5,
         color='green' if pjb_g > 0.05 else 'red',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
ax1.grid(linestyle='--', alpha=0.3)

# ── B. ACF résidus gazole ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
lags = np.arange(1, 13)
ci   = 1.96 / np.sqrt(nobs)
ax2.bar(lags, acf_g, color=C1, alpha=0.7, width=0.6)
ax2.axhline( ci, color='red', linestyle='--', linewidth=1, label=f'IC 95% (±{ci:.3f})')
ax2.axhline(-ci, color='red', linestyle='--', linewidth=1)
ax2.axhline(0,   color='black', linewidth=0.8)
ax2.set_title('Test blancheur — ACF résidus (gazole)', fontsize=10, fontweight='bold')
ax2.set_xlabel('Lag (mois)')
ax2.set_ylabel('Autocorrélation')
verdict_lb_g = "✓ Blancs" if pQ_g > 0.05 else "✗ Autocorrélés"
ax2.text(0.05, 0.92, f'Ljung-Box Q p={pQ_g:.4f}  {verdict_lb_g}',
         transform=ax2.transAxes, fontsize=8.5,
         color='green' if pQ_g > 0.05 else 'red',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
ax2.legend(fontsize=8)
ax2.grid(axis='y', linestyle='--', alpha=0.3)

# ── C. QQ-plot résidus Brent ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
(osm2, osr2), (slope2, intercept2, r2) = stats.probplot(resid_brent)
ax3.plot(osm2, osr2, 'o', color=C2, markersize=3, alpha=0.6, label='Résidus Brent')
ax3.plot(osm2, slope2*np.array(osm2)+intercept2, '--', color='black', linewidth=1.2)
ax3.set_title('Test normalité — QQ-plot (Brent)', fontsize=10, fontweight='bold')
ax3.set_xlabel('Quantiles théoriques (loi normale)')
ax3.set_ylabel('Quantiles observés')
verdict_jb_b = "✓ Normale" if pjb_b > 0.05 else "✗ Non normale"
ax3.text(0.05, 0.92, f'Jarque-Bera p={pjb_b:.4f}  {verdict_jb_b}',
         transform=ax3.transAxes, fontsize=8.5,
         color='green' if pjb_b > 0.05 else 'red',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
ax3.grid(linestyle='--', alpha=0.3)

# ── D. Racines du VAR (stabilité) ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
theta = np.linspace(0, 2 * np.pi, 300)
ax4.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.2, label='Cercle unité')
ax4.scatter(eigenvalues.real, eigenvalues.imag,
            color=[C1 if m <= 1.0 else 'red' for m in moduli],
            s=60, zorder=5, label='Valeurs propres')
ax4.axhline(0, color='gray', linewidth=0.6)
ax4.axvline(0, color='gray', linewidth=0.6)
ax4.set_xlim(-1.3, 1.3)
ax4.set_ylim(-1.3, 1.3)
ax4.set_aspect('equal')
ax4.set_title('Test stabilité — racines du VAR companion', fontsize=10, fontweight='bold')
ax4.set_xlabel('Partie réelle')
ax4.set_ylabel('Partie imaginaire')
verdict_stab = "✓ Stable" if stable else "✗ Instable"
ax4.text(0.05, 0.92, f'Toutes les racines ≤ 1  {verdict_stab}',
         transform=ax4.transAxes, fontsize=8.5,
         color='green' if stable else 'red',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
ax4.legend(fontsize=8)
ax4.grid(linestyle='--', alpha=0.3)

plt.savefig('graphique_validation.png', dpi=150, bbox_inches='tight')
print("✓ Graphique sauvegardé : graphique_validation.png")

# =============================================================================
# 6. FICHIER TEXTE DE SYNTHÈSE
# =============================================================================
lines = []
lines.append("VALIDATION DU VECM — Synthèse des 3 tests essentiels")
lines.append("=" * 60)

lines.append("\n1. TEST DE NORMALITÉ DES RÉSIDUS (Jarque-Bera)")
lines.append("   H0 : les résidus suivent une loi normale")
lines.append(f"   Gazole  : stat={jb_g:.3f}, p={pjb_g:.4f} → {'✓ On ne rejette pas H0 (normale)' if pjb_g>0.05 else '✗ On rejette H0 (non normale)'}")
lines.append(f"             asymétrie={skew_g:.3f}, excès de kurtosis={kurt_g:.3f}")
lines.append(f"   Brent   : stat={jb_b:.3f}, p={pjb_b:.4f} → {'✓ On ne rejette pas H0 (normale)' if pjb_b>0.05 else '✗ On rejette H0 (non normale)'}")
lines.append(f"             asymétrie={skew_b:.3f}, excès de kurtosis={kurt_b:.3f}")
if pjb_g <= 0.05 or pjb_b <= 0.05:
    lines.append("   → Résidus non normaux : fréquent sur des séries de prix avec")
    lines.append("     ruptures structurelles (2008, 2020, 2022). Les intervalles")
    lines.append("     de confiance des prévisions sont à interpréter avec prudence.")

lines.append("\n2. TEST DE BLANCHEUR DES RÉSIDUS (Ljung-Box, 12 lags)")
lines.append("   H0 : les résidus ne sont pas autocorrélés")
lines.append(f"   Gazole  : Q={Q_g:.3f}, p={pQ_g:.4f} → {'✓ Résidus blancs' if pQ_g>0.05 else '✗ Autocorrélation résiduelle'}")
lines.append(f"   Brent   : Q={Q_b:.3f}, p={pQ_b:.4f} → {'✓ Résidus blancs' if pQ_b>0.05 else '✗ Autocorrélation résiduelle'}")
if pQ_g <= 0.05 or pQ_b <= 0.05:
    lines.append("   → Autocorrélation détectée : le modèle ne capture pas toute")
    lines.append("     la dynamique. Essayer k=2 lags pourrait améliorer le fit.")

lines.append("\n3. TEST DE STABILITÉ (racines de la matrice companion)")
lines.append("   H0 : toutes les valeurs propres sont dans le cercle unité")
lines.append(f"   Modules des valeurs propres : {[f'{m:.4f}' for m in sorted(moduli, reverse=True)]}")
lines.append(f"   → {'✓ Modèle stable : les prévisions convergent' if stable else '✗ Modèle instable : prévisions divergentes'}")

lines.append("\n" + "=" * 60)
lines.append("CONCLUSION GLOBALE")
n_ok = sum([pjb_g>0.05, pjb_b>0.05, pQ_g>0.05, pQ_b>0.05, stable])
lines.append(f"  {n_ok}/5 critères validés.")
if n_ok >= 4:
    lines.append("  → Le modèle est globalement satisfaisant.")
elif n_ok >= 2:
    lines.append("  → Résultats acceptables mais à nuancer dans le rapport.")
else:
    lines.append("  → Modèle fragile — résultats à interpréter avec précaution.")

output = "\n".join(lines)
print("\n" + output)
with open("validation_vecm.txt", "w", encoding="utf-8") as f:
    f.write(output)
print("\n✓ Fichier créé : validation_vecm.txt")
plt.show()
