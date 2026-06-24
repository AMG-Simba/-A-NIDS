"""
Entraînement de l'Agent IA — A-NIDS
====================================
Entraîne et compare plusieurs modèles ML sur le VRAI dataset NSL-KDD.
L'agent IA retenu sera comparé au détecteur classique (rule_based_ids.py).

Modèles comparés :
- Logistic Regression (baseline linéaire)
- Decision Tree
- Random Forest
- Gradient Boosting
- MLP (réseau de neurones)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

from data_loader import load_nsl_kdd

BASE = Path(__file__).parent.parent
MODELS = BASE / "models"
RESULTS = BASE / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

CATEGORICAL = ['protocol_type', 'service', 'flag']


def preprocess(df_train, df_test):
    """Encode + normalise les données. Retourne X/y train et test."""
    print("\n[1/5] Prétraitement des données réelles...")

    # Colonnes à exclure des features
    drop_cols = ['label', 'attack_category', 'binary_label']

    # Encodage des variables catégorielles (fit sur train+test pour cohérence)
    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        combined = pd.concat([df_train[col], df_test[col]], axis=0)
        le.fit(combined)
        df_train[col] = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        encoders[col] = le

    feature_names = [c for c in df_train.columns if c not in drop_cols]

    X_train = df_train[feature_names].values
    y_train = df_train['binary_label'].values
    X_test = df_test[feature_names].values
    y_test = df_test['binary_label'].values

    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Sauvegarde des préprocesseurs
    joblib.dump(scaler, MODELS / "scaler.pkl")
    joblib.dump(encoders, MODELS / "encoders.pkl")
    joblib.dump(feature_names, MODELS / "feature_names.pkl")

    print(f"  Train : {X_train.shape[0]} connexions, {X_train.shape[1]} features")
    print(f"  Test  : {X_test.shape[0]} connexions")
    print(f"  (Le test contient des attaques INCONNUES à l'entraînement)")

    return X_train, y_train, X_test, y_test, feature_names


def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20,
                                                 random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=6,
                                                        random_state=42),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(max_iter=300, max_depth=12,
                                                                 learning_rate=0.1, random_state=42),
        'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                                              random_state=42, early_stopping=True),
    }


def train_and_evaluate(X_train, y_train, X_test, y_test):
    print("\n[2/5] Entraînement des modèles IA sur données réelles...")
    results = {}

    for name, model in get_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - t0

        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            'train_time': train_time,
            'pred_time': pred_time,
        }
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {'model': model, 'metrics': metrics, 'cm': cm,
                         'y_pred': y_pred, 'y_proba': y_proba}

        joblib.dump(model, MODELS / f"{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.pkl")
        print(f"  {name:<24} F1={metrics['f1_score']:.4f}  "
              f"Acc={metrics['accuracy']:.4f}  ({train_time:.1f}s)")

    return results


def plot_comparison(results, y_test):
    print("\n[3/5] Génération des graphiques...")

    # 1. Comparaison des métriques
    metrics_df = pd.DataFrame({
        name: [r['metrics']['accuracy'], r['metrics']['precision'],
               r['metrics']['recall'], r['metrics']['f1_score']]
        for name, r in results.items()
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df.T.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black')
    ax.set_title('Comparaison des modèles IA — Dataset réel NSL-KDD',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right'); ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=20, ha='right'); plt.tight_layout()
    plt.savefig(RESULTS / "comparaison_modeles.png", dpi=120, bbox_inches='tight')
    plt.close()

    # 2. Matrices de confusion
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    for ax, (name, r) in zip(axes, results.items()):
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Attaque'], yticklabels=['Normal', 'Attaque'], cbar=False)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_ylabel('Réel'); ax.set_xlabel('Prédit')
    plt.suptitle('Matrices de confusion — NSL-KDD', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS / "matrices_confusion.png", dpi=120, bbox_inches='tight')
    plt.close()

    # 3. Courbes ROC
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, r in results.items():
        if r['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
            ax.plot(fpr, tpr, label=f"{name} (AUC={r['metrics']['roc_auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Hasard')
    ax.set_xlabel('Taux de faux positifs'); ax.set_ylabel('Taux de vrais positifs')
    ax.set_title('Courbes ROC — NSL-KDD', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(RESULTS / "courbes_roc.png", dpi=120, bbox_inches='tight')
    plt.close()

    # 4. Importance des features (meilleur modèle arbre)
    best_tree = results.get('Random Forest', {}).get('model')
    if best_tree is not None and hasattr(best_tree, 'feature_importances_'):
        fnames = joblib.load(MODELS / "feature_names.pkl")
        imp = pd.Series(best_tree.feature_importances_, index=fnames).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(10, 7))
        imp.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title('Top 15 features — Random Forest (NSL-KDD)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance'); ax.grid(axis='x', alpha=0.3); plt.tight_layout()
        plt.savefig(RESULTS / "importance_features.png", dpi=120, bbox_inches='tight')
        plt.close()

    print("  4 graphiques générés.")


def save_results(results):
    print("\n[4/5] Sauvegarde des résultats...")
    best_name = max(results, key=lambda n: results[n]['metrics']['f1_score'])

    json_out = {}
    for name, r in results.items():
        m = dict(r['metrics'])
        json_out[name] = {'metrics': m, 'confusion_matrix': r['cm'].tolist()}
    json_out['best_model'] = best_name

    (RESULTS / "metrics.json").write_text(json.dumps(json_out, indent=2, ensure_ascii=False))

    # Sauver le meilleur modèle
    joblib.dump(results[best_name]['model'], MODELS / "best_model.pkl")
    (MODELS / "best_model_name.txt").write_text(best_name)

    print(f"  Meilleur agent IA : {best_name} (F1={results[best_name]['metrics']['f1_score']:.4f})")
    return best_name


def main():
    print("=" * 65)
    print("ENTRAÎNEMENT DE L'AGENT IA — A-NIDS (dataset réel NSL-KDD)")
    print("=" * 65)

    df_train = load_nsl_kdd('train')
    df_test = load_nsl_kdd('test')

    X_train, y_train, X_test, y_test, fnames = preprocess(df_train, df_test)
    results = train_and_evaluate(X_train, y_train, X_test, y_test)
    plot_comparison(results, y_test)
    best = save_results(results)

    print("\n[5/5] Terminé.")
    print("=" * 65)
    # Tableau final
    print(f"\n{'Modèle':<24}{'Accuracy':<11}{'Precision':<11}{'Recall':<10}{'F1':<9}{'AUC':<8}")
    print("-" * 73)
    for name, r in results.items():
        m = r['metrics']
        auc = f"{m['roc_auc']:.4f}" if m['roc_auc'] else "N/A"
        star = " *" if name == best else ""
        print(f"{name:<24}{m['accuracy']:<11.4f}{m['precision']:<11.4f}"
              f"{m['recall']:<10.4f}{m['f1_score']:<9.4f}{auc:<8}{star}")


if __name__ == "__main__":
    main()
