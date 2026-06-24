"""
COMPARAISON : Agent IA  vs  Détecteur Classique à signatures
=============================================================
C'est la démonstration centrale du projet. On compare sur le MÊME jeu de
test réel (NSL-KDD) :
  - l'AGENT IA (meilleur modèle ML entraîné)
  - le DÉTECTEUR CLASSIQUE (règles fixes type Snort)

On analyse en particulier la détection des ATTAQUES INCONNUES — là où
l'IA démontre sa supériorité sur les approches à signatures.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

from data_loader import load_nsl_kdd, ATTACK_MAP
from rule_based_ids import RuleBasedIDS

BASE = Path(__file__).parent.parent
MODELS = BASE / "models"
RESULTS = BASE / "results"

CATEGORICAL = ['protocol_type', 'service', 'flag']


def evaluate_ai(df_test):
    """Évalue l'agent IA sur le jeu de test."""
    model = joblib.load(MODELS / "best_model.pkl")
    scaler = joblib.load(MODELS / "scaler.pkl")
    encoders = joblib.load(MODELS / "encoders.pkl")
    fnames = joblib.load(MODELS / "feature_names.pkl")
    name = (MODELS / "best_model_name.txt").read_text().strip()

    df = df_test.copy()
    for col, le in encoders.items():
        df[col] = le.transform(df[col])
    X = scaler.transform(df[fnames].values)
    y_pred = model.predict(X)
    return y_pred, name


def metrics_dict(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }


def recall_by_category(df_test, y_pred):
    """Calcule le taux de détection par catégorie d'attaque."""
    df = df_test.copy()
    df['pred'] = y_pred
    out = {}
    for cat in ['DoS', 'Probe', 'R2L', 'U2R']:
        mask = df['attack_category'] == cat
        if mask.sum() > 0:
            # parmi les attaques de cette catégorie, combien détectées (pred=1)
            out[cat] = df.loc[mask, 'pred'].mean()
    return out


def main():
    print("=" * 70)
    print("COMPARAISON : AGENT IA  vs  DÉTECTEUR CLASSIQUE (signatures)")
    print("=" * 70)

    df_test = load_nsl_kdd('test')
    y_true = df_test['binary_label'].values

    # --- Agent IA ---
    print("\n[1] Évaluation de l'agent IA...")
    y_ai, ai_name = evaluate_ai(df_test)
    m_ai = metrics_dict(y_true, y_ai)
    rec_ai = recall_by_category(df_test, y_ai)

    # --- Détecteur classique ---
    print("[2] Évaluation du détecteur classique (signatures)...")
    ids = RuleBasedIDS()
    y_rule, _ = ids.predict(df_test)
    m_rule = metrics_dict(y_true, y_rule)
    rec_rule = recall_by_category(df_test, y_rule)

    # --- Tableau comparatif ---
    print("\n" + "=" * 70)
    print("RÉSULTATS COMPARATIFS (jeu de test réel = 22 544 connexions)")
    print("=" * 70)
    print(f"\n{'Métrique':<16}{'Classique (règles)':<22}{'Agent IA (' + ai_name + ')':<26}{'Gain IA':<10}")
    print("-" * 70)
    for key, label in [('accuracy', 'Accuracy'), ('precision', 'Precision'),
                       ('recall', 'Recall'), ('f1_score', 'F1-Score')]:
        gain = m_ai[key] - m_rule[key]
        sign = '+' if gain >= 0 else ''
        print(f"{label:<16}{m_rule[key]:<22.4f}{m_ai[key]:<26.4f}{sign}{gain:.4f}")

    # --- Détection par catégorie ---
    print(f"\n{'Catégorie':<14}{'Détection Classique':<22}{'Détection IA':<16}{'Gain':<10}")
    print("-" * 62)
    for cat in ['DoS', 'Probe', 'R2L', 'U2R']:
        rc = rec_rule.get(cat, 0)
        ra = rec_ai.get(cat, 0)
        gain = ra - rc
        sign = '+' if gain >= 0 else ''
        print(f"{cat:<14}{rc:<22.2%}{ra:<16.2%}{sign}{gain:.2%}")

    # --- Graphique 1 : métriques globales ---
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    rule_vals = [m_rule['accuracy'], m_rule['precision'], m_rule['recall'], m_rule['f1_score']]
    ai_vals = [m_ai['accuracy'], m_ai['precision'], m_ai['recall'], m_ai['f1_score']]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w/2, rule_vals, w, label='Détecteur Classique (signatures)',
           color='#94a3b8', edgecolor='black')
    ax.bar(x + w/2, ai_vals, w, label=f'Agent IA ({ai_name})',
           color='#0ea5e9', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
    ax.set_title('IA vs Classique — Performances globales (NSL-KDD réel)',
                 fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for i, (r, a) in enumerate(zip(rule_vals, ai_vals)):
        ax.text(i - w/2, r + 0.02, f'{r:.2f}', ha='center', fontsize=9)
        ax.text(i + w/2, a + 0.02, f'{a:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS / "comparaison_ia_vs_classique.png", dpi=120, bbox_inches='tight')
    plt.close()

    # --- Graphique 2 : détection par catégorie ---
    fig, ax = plt.subplots(figsize=(10, 6))
    cats = ['DoS', 'Probe', 'R2L', 'U2R']
    rule_c = [rec_rule.get(c, 0) for c in cats]
    ai_c = [rec_ai.get(c, 0) for c in cats]
    x = np.arange(len(cats))
    ax.bar(x - w/2, rule_c, w, label='Détecteur Classique', color='#94a3b8', edgecolor='black')
    ax.bar(x + w/2, ai_c, w, label='Agent IA', color='#0ea5e9', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Taux de détection (recall)')
    ax.set_title("Taux de détection par type d'attaque — IA vs Classique",
                 fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for i, (r, a) in enumerate(zip(rule_c, ai_c)):
        ax.text(i - w/2, r + 0.02, f'{r:.0%}', ha='center', fontsize=9)
        ax.text(i + w/2, a + 0.02, f'{a:.0%}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS / "detection_par_categorie.png", dpi=120, bbox_inches='tight')
    plt.close()

    # --- Sauvegarde JSON ---
    comparison = {
        'ai_model': ai_name,
        'ai_metrics': m_ai,
        'rule_metrics': m_rule,
        'ai_recall_by_category': rec_ai,
        'rule_recall_by_category': rec_rule,
        'test_size': len(df_test),
    }
    (RESULTS / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"  L'agent IA détecte {m_ai['recall']:.1%} des attaques contre "
          f"{m_rule['recall']:.1%} pour le détecteur classique.")
    print(f"  Gain de F1-Score : +{(m_ai['f1_score'] - m_rule['f1_score']):.1%}")
    print(f"  2 graphiques + comparison.json générés dans results/")


if __name__ == "__main__":
    main()
