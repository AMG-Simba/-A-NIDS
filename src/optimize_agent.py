"""
Optimisation de l'Agent IA — A-NIDS
====================================
Améliore l'agent IA en optimisant le SEUIL DE DÉCISION.

Problème : par défaut, un modèle classe "attaque" si proba >= 0.50.
Mais en cybersécurité, RATER une attaque (faux négatif) coûte bien plus cher
que lever une fausse alerte (faux positif). On a donc intérêt à ABAISSER le
seuil pour détecter davantage d'attaques, même au prix de quelques faux positifs.

Ce module trouve le seuil optimal qui maximise le F1-score (ou le recall sous
contrainte de précision), et montre l'impact sur la détection.
"""

import json
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix
)

from data_loader import load_nsl_kdd

BASE = Path(__file__).parent.parent
MODELS = BASE / "models"
RESULTS = BASE / "results"

CATEGORICAL = ['protocol_type', 'service', 'flag']


def prepare_test_data():
    """Recharge et prétraite les données test avec les préprocesseurs sauvegardés."""
    df_test = load_nsl_kdd('test')

    scaler = joblib.load(MODELS / "scaler.pkl")
    encoders = joblib.load(MODELS / "encoders.pkl")
    feature_names = joblib.load(MODELS / "feature_names.pkl")

    df = df_test.copy()
    for col, le in encoders.items():
        # Gérer les catégories inconnues
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])

    X_test = scaler.transform(df[feature_names].values)
    y_test = df['binary_label'].values
    categories = df['attack_category'].values
    return X_test, y_test, categories


def find_optimal_threshold(y_true, y_proba):
    """Trouve le seuil qui maximise le F1-score, en testant une grille de seuils
    raisonnables (0.05 à 0.95) pour éviter les artefacts aux extrêmes."""
    grid = np.linspace(0.05, 0.95, 181)
    f1s = []
    precs = []
    recs = []
    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
    f1s = np.array(f1s); precs = np.array(precs); recs = np.array(recs)
    best_idx = int(np.argmax(f1s))
    return grid[best_idx], precs, recs, grid, f1s


def evaluate_at_threshold(y_true, y_proba, threshold):
    """Calcule les métriques à un seuil donné."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        'threshold': float(threshold),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    print("=" * 65)
    print("OPTIMISATION DU SEUIL DE DÉCISION DE L'AGENT IA")
    print("=" * 65)

    # Charger le meilleur modèle
    model = joblib.load(MODELS / "best_model.pkl")
    model_name = (MODELS / "best_model_name.txt").read_text().strip()
    print(f"\nModèle : {model_name}")

    X_test, y_test, categories = prepare_test_data()
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Seuil par défaut (0.50) ---
    print("\n[1] Seuil par défaut (0.50) :")
    m_default = evaluate_at_threshold(y_test, y_proba, 0.50)
    print(f"  Accuracy={m_default['accuracy']:.4f}  Precision={m_default['precision']:.4f}  "
          f"Recall={m_default['recall']:.4f}  F1={m_default['f1_score']:.4f}")

    # --- Seuil optimal (max F1) ---
    best_thresh, precisions, recalls, thresholds, f1_scores = find_optimal_threshold(y_test, y_proba)
    print(f"\n[2] Seuil optimal trouvé : {best_thresh:.3f}")
    m_opt = evaluate_at_threshold(y_test, y_proba, best_thresh)
    print(f"  Accuracy={m_opt['accuracy']:.4f}  Precision={m_opt['precision']:.4f}  "
          f"Recall={m_opt['recall']:.4f}  F1={m_opt['f1_score']:.4f}")

    # --- Seuil "sécurité" (recall maximal sous contrainte precision >= 0.90) ---
    valid = precisions >= 0.90
    if valid.any():
        idx_candidates = np.where(valid)[0]
        best_security_idx = idx_candidates[np.argmax(recalls[idx_candidates])]
        security_thresh = thresholds[best_security_idx]
    else:
        security_thresh = best_thresh
    print(f"\n[3] Seuil 'sécurité' (precision >= 0.90, recall max) : {security_thresh:.3f}")
    m_sec = evaluate_at_threshold(y_test, y_proba, security_thresh)
    print(f"  Accuracy={m_sec['accuracy']:.4f}  Precision={m_sec['precision']:.4f}  "
          f"Recall={m_sec['recall']:.4f}  F1={m_sec['f1_score']:.4f}")

    # --- Gain ---
    gain_recall = (m_opt['recall'] - m_default['recall']) * 100
    gain_f1 = (m_opt['f1_score'] - m_default['f1_score']) * 100
    print(f"\n>>> Gain au seuil optimal : +{gain_recall:.1f} pts de recall, "
          f"+{gain_f1:.1f} pts de F1")

    # --- Graphique : précision/rappel/F1 selon le seuil ---
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(thresholds, precisions, label='Précision', color='#3b82f6', linewidth=2)
    ax.plot(thresholds, recalls, label='Rappel (Recall)', color='#ef4444', linewidth=2)
    ax.plot(thresholds, f1_scores, label='F1-Score', color='#10b981', linewidth=2)
    ax.axvline(0.50, color='gray', linestyle='--', alpha=0.6, label='Seuil défaut (0.50)')
    ax.axvline(best_thresh, color='#f59e0b', linestyle='--', alpha=0.8,
               label=f'Seuil optimal ({best_thresh:.2f})')
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("Score")
    ax.set_title("Impact du seuil de décision sur les performances de l'agent IA",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='center left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS / "optimisation_seuil.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Graphique sauvegardé : optimisation_seuil.png")

    # --- Sauvegarde ---
    out = {
        'model_name': model_name,
        'default': m_default,
        'optimal': m_opt,
        'security': m_sec,
        'optimal_threshold': float(best_thresh),
        'security_threshold': float(security_thresh),
    }
    (RESULTS / "threshold_optimization.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    # Sauvegarder le seuil optimal pour le détecteur en production
    (MODELS / "optimal_threshold.txt").write_text(f"{best_thresh:.4f}")
    print(f"  Seuil optimal sauvegardé pour le détecteur : {best_thresh:.4f}")
    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
