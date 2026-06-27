"""
simulation_choc_paix.py
=======================
Simulation de l'impact d'un choc de paix au Moyen-Orient sur les prix du gazole en France.
Un graphique par scénario, sauvegardé séparément.

Méthode : propagation via les coefficients VECM estimés (Annexe 4).
Paramètres issus directement des résultats du rapport MODS Gazole.

Auteurs : Philibert De Brabois, Soline Carle — Télécom Paris, 2026
"""

import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. PARAMÈTRES ESTIMÉS (tirés des annexes)
# ──────────────────────────────────────────────

# Coefficients VECM symétrique (Annexe 4) — vitesse de convergence vers nouvel équilibre
ALPHA_BRENT_HT  = 0.0604   # demi-vie 11,1 semaines
ALPHA_BRENT_TTC = 0.0318   # demi-vie 21,4 semaines

# Élasticités de long terme (Annexe 3 / Johansen)
ELAS_HT  = 0.885
ELAS_TTC = 0.540

# Niveaux initiaux
PRIX_BRENT_INITIAL   = 85.0
PRIX_GAZOLE_HT_INIT  = 0.863
PRIX_GAZOLE_TTC_INIT = 1.95

# Planchers fiscaux
PLANCHER_TTC = 0.608 * 1.20   # TICPE + TVA sur TICPE ≈ 0.73 €/L
PLANCHER_HT  = 0.608          # TICPE seule, sans TVA

# ──────────────────────────────────────────────
# 2. SCÉNARIOS
# ──────────────────────────────────────────────

SCENARIOS = [
    {"label": "Scénario 1 — Désescalade partielle",  "sous_titre": "Choc modéré : Brent −15%",    "choc": -0.15, "couleur_fond": "#eaf4fb"},
    {"label": "Scénario 2 — Fin de la guerre",        "sous_titre": "Choc fort : Brent −25%",      "choc": -0.25, "couleur_fond": "#eafaf1"},
    {"label": "Scénario 3 — Paix + levée sanctions",  "sous_titre": "Choc majeur : Brent −35%",    "choc": -0.35, "couleur_fond": "#fdf2f8"},
]

N_SEMAINES = 260  # 5 ans pour montrer la convergence lente

# ──────────────────────────────────────────────
# 3. SIMULATION
# ──────────────────────────────────────────────

def simuler_trajectoire(choc_pct, n_semaines=N_SEMAINES):
    brent_init  = PRIX_BRENT_INITIAL
    brent_cible = PRIX_BRENT_INITIAL * (1 + choc_pct)

    brent   = np.zeros(n_semaines)
    gaz_ht  = np.zeros(n_semaines)
    gaz_ttc = np.zeros(n_semaines)

    brent[0]   = brent_init
    gaz_ht[0]  = PRIX_GAZOLE_HT_INIT
    gaz_ttc[0] = PRIX_GAZOLE_TTC_INIT

    # Cibles d'équilibre de long terme via élasticité
    ratio = brent_cible / brent_init
    gaz_ht_cible  = max(PRIX_GAZOLE_HT_INIT  * (ratio ** ELAS_HT),  PLANCHER_HT)
    gaz_ttc_cible = max(PRIX_GAZOLE_TTC_INIT * (ratio ** ELAS_TTC), PLANCHER_TTC)

    for t in range(n_semaines - 1):
        # Brent converge exponentiellement vers sa nouvelle cible (VECM HT)
        brent[t+1] = brent_cible + (brent[t] - brent_cible) * (1 - ALPHA_BRENT_HT)

        # Gazole HT suit avec inertie propre (même vitesse que Brent HT)
        gaz_ht[t+1] = max(
            gaz_ht_cible + (gaz_ht[t] - gaz_ht_cible) * (1 - ALPHA_BRENT_HT),
            PLANCHER_HT
        )

        # Gazole TTC plus lent à cause du filtre fiscal (VECM TTC)
        gaz_ttc[t+1] = max(
            gaz_ttc_cible + (gaz_ttc[t] - gaz_ttc_cible) * (1 - ALPHA_BRENT_TTC),
            PLANCHER_TTC
        )

    return {
        "semaines": np.arange(n_semaines),
        "brent": brent,
        "gaz_ht": gaz_ht,
        "gaz_ttc": gaz_ttc,
        "gaz_ht_cible": gaz_ht_cible,
        "gaz_ttc_cible": gaz_ttc_cible,
    }


def calculer_demi_vie(traj, prix_init, prix_cible, serie):
    seuil = prix_init - 0.5 * (prix_init - prix_cible)
    for i, v in enumerate(traj[serie]):
        if v <= seuil:
            return i
    return None

# ──────────────────────────────────────────────
# 4. AXE TEMPOREL (5 ans)
# ──────────────────────────────────────────────

SEMAINES_AXE = np.arange(0, N_SEMAINES + 1, 26)
LABELS_AXE   = ["Choc"] + [f"S+{s}\n({s//4} mois)" for s in SEMAINES_AXE[1:]]

# ──────────────────────────────────────────────
# 5. UN GRAPHIQUE PAR SCÉNARIO
# ──────────────────────────────────────────────

for sc in SCENARIOS:
    choc = sc["choc"]
    traj = simuler_trajectoire(choc)
    semaines = traj["semaines"]

    gaz_ht_cible  = traj["gaz_ht_cible"]
    gaz_ttc_cible = traj["gaz_ttc_cible"]

    brent_normalise = traj["brent"] / PRIX_BRENT_INITIAL * PRIX_GAZOLE_TTC_INIT

    dv_ttc = calculer_demi_vie(traj, PRIX_GAZOLE_TTC_INIT, gaz_ttc_cible, "gaz_ttc")
    dv_ht  = calculer_demi_vie(traj, PRIX_GAZOLE_HT_INIT,  gaz_ht_cible,  "gaz_ht")

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(sc["couleur_fond"])
    ax.set_facecolor(sc["couleur_fond"])

    # ── Zone de choc ──
    ax.axvspan(0, 1.5, color="#e74c3c", alpha=0.12, zorder=0)
    ax.text(0.75, PRIX_GAZOLE_TTC_INIT + 0.06, "Choc", ha="center",
            fontsize=8, color="#c0392b", fontstyle="italic")

    # ── Brent normalisé ──
    ax.plot(semaines, brent_normalise, color="#95a5a6", lw=1.5, ls="--",
            alpha=0.6, label="Brent normalisé à l'échelle gazole (référence)")

    # ── Gazole HT ──
    ax.plot(semaines, traj["gaz_ht"], color="#117a65", lw=2.5,
            label="Gazole HT — réalité industrielle")

    # ── Gazole TTC ──
    ax.plot(semaines, traj["gaz_ttc"], color="#1a5276", lw=3,
            label="Gazole TTC — prix à la pompe")

    # ── Planchers fiscaux ──
    ax.axhline(PLANCHER_TTC, color="#c0392b", lw=1.5, ls=":",
               label=f"Plancher fiscal TTC ({PLANCHER_TTC:.2f} €/L)")
    ax.axhline(PLANCHER_HT, color="#117a65", lw=1.0, ls=":",
               alpha=0.5, label=f"Plancher fiscal HT — TICPE seule ({PLANCHER_HT:.3f} €/L)")

    # ── Lignes horizontales prix initiaux / cibles ──
    ax.axhline(PRIX_GAZOLE_TTC_INIT, color="#1a5276", lw=0.8, ls="--", alpha=0.25)
    ax.axhline(gaz_ttc_cible,        color="#1a5276", lw=0.8, ls="-.", alpha=0.35)
    ax.axhline(PRIX_GAZOLE_HT_INIT,  color="#117a65", lw=0.8, ls="--", alpha=0.25)
    ax.axhline(gaz_ht_cible,         color="#117a65", lw=0.8, ls="-.", alpha=0.35)

    # ── Annotations prix ──
    ax.annotate(f"{PRIX_GAZOLE_TTC_INIT:.2f} €/L  (pré-choc TTC)",
                xy=(2, PRIX_GAZOLE_TTC_INIT), xytext=(5, PRIX_GAZOLE_TTC_INIT + 0.025),
                fontsize=9, color="#1a5276", va="bottom")
    ax.annotate(f"{gaz_ttc_cible:.2f} €/L  (équilibre cible TTC)",
                xy=(N_SEMAINES - 5, gaz_ttc_cible),
                xytext=(N_SEMAINES - 80, gaz_ttc_cible + 0.025),
                fontsize=9, color="#1a5276", alpha=0.8, va="bottom")
    ax.annotate(f"{PRIX_GAZOLE_HT_INIT:.3f} €/L  (pré-choc HT)",
                xy=(2, PRIX_GAZOLE_HT_INIT), xytext=(5, PRIX_GAZOLE_HT_INIT - 0.04),
                fontsize=9, color="#117a65", va="top")
# Décaler légèrement l'annotation cible HT pour qu'elle ne chevauche pas le plancher
    ax.annotate(f"{gaz_ht_cible:.3f} €/L  (équilibre cible HT — limité par TICPE)",
                xy=(N_SEMAINES - 5, gaz_ht_cible),
                xytext=(N_SEMAINES - 120, gaz_ht_cible + 0.04),  # au-dessus au lieu de en-dessous
                fontsize=9, color="#117a65", alpha=0.8, va="bottom")

    # ── Demi-vies ──
    if dv_ttc:
        ax.axvline(dv_ttc, color="#1a5276", lw=1, ls=":", alpha=0.6)
        ax.annotate(
            f"Demi-vie TTC\nS+{dv_ttc} ({dv_ttc//4} mois)",
            xy=(dv_ttc, (PRIX_GAZOLE_TTC_INIT + gaz_ttc_cible) / 2),
            xytext=(dv_ttc + 6, (PRIX_GAZOLE_TTC_INIT + gaz_ttc_cible) / 2 + 0.05),
            fontsize=9, color="#1a5276", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1a5276", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1a5276", alpha=0.8)
        )

    if dv_ht:
        ax.axvline(dv_ht, color="#117a65", lw=1, ls=":", alpha=0.6)
        ax.annotate(
            f"Demi-vie HT\nS+{dv_ht} ({dv_ht//4} mois)",
            xy=(dv_ht, (PRIX_GAZOLE_HT_INIT + gaz_ht_cible) / 2),
            xytext=(dv_ht + 6, (PRIX_GAZOLE_HT_INIT + gaz_ht_cible) / 2 - 0.07),
            fontsize=9, color="#117a65", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#117a65", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#117a65", alpha=0.8)
        )

    # ── Encart récapitulatif ──
    baisse_ttc = gaz_ttc_cible - PRIX_GAZOLE_TTC_INIT
    baisse_ht  = gaz_ht_cible  - PRIX_GAZOLE_HT_INIT
    recap = (
        f"Brent : {choc*100:+.0f}%  ({PRIX_BRENT_INITIAL:.0f} → {PRIX_BRENT_INITIAL*(1+choc):.0f} €/baril)\n"
        f"Gazole TTC : {baisse_ttc:+.3f} €/L  ({PRIX_GAZOLE_TTC_INIT:.2f} → {gaz_ttc_cible:.2f} €/L)\n"
        f"Gazole HT  : {baisse_ht:+.3f} €/L  ({PRIX_GAZOLE_HT_INIT:.3f} → {gaz_ht_cible:.3f} €/L)\n"
        f"Plancher TTC : {PLANCHER_TTC:.2f} €/L  |  Plancher HT : {PLANCHER_HT:.3f} €/L"
    )
    ax.text(0.98, 0.97, recap, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bdc3c7", alpha=0.92),
            family="monospace")

    # ── Mise en forme ──
    ax.set_xlim(0, N_SEMAINES)
    ax.set_ylim(
        min(gaz_ht_cible, PLANCHER_HT) - 0.08,
        PRIX_GAZOLE_TTC_INIT + 0.15
    )
    ax.set_xticks(SEMAINES_AXE)
    ax.set_xticklabels(LABELS_AXE, fontsize=8)
    ax.set_ylabel("Prix (€/L)", fontsize=11)
    ax.set_xlabel("Semaines après le choc de paix", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", lw=0.5, alpha=0.4, color="#bdc3c7")
    ax.grid(axis="x", lw=0.3, alpha=0.2, color="#bdc3c7")
    ax.spines[["top", "right"]].set_visible(False)

    ax.set_title(
        f"{sc['label']}\n{sc['sous_titre']}",
        fontsize=13, fontweight="bold", pad=12, color="#2c3e50"
    )
    fig.text(0.5, 0.01,
             "Méthode : VECM symétrique (demi-vie Brent HT = 11,1 sem., TTC = 21,4 sem.) — Coefficients estimés sur données hebdomadaires 2005–2026",
             ha="center", fontsize=8, color="#7f8c8d", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    nom_fichier = f"simulation_scenario_{abs(int(choc*100))}pct.png"
    plt.savefig(nom_fichier, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"✅ Sauvegardé : {nom_fichier}")
    plt.show()
    plt.close()

print("\nTerminé — 3 figures générées.")