"""
Détecteur CLASSIQUE à base de signatures / règles
===================================================
Simule un IDS traditionnel type Snort/Suricata : il détecte les attaques
grâce à un ensemble de RÈGLES FIXES écrites à la main par des experts,
basées sur les signatures d'attaques CONNUES.

C'est le système qu'on compare à l'agent IA. Sa faiblesse fondamentale :
il ne détecte que ce qui correspond à ses règles → rate les attaques inconnues.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import load_nsl_kdd


class RuleBasedIDS:
    """IDS classique à signatures (équivalent simplifié de Snort/Suricata)."""

    def __init__(self):
        # Règles de détection écrites à la main par des "experts sécurité".
        # Chaque règle cible une signature d'attaque connue.
        self.rules = [
            self._rule_syn_flood,
            self._rule_port_scan,
            self._rule_brute_force,
            self._rule_icmp_flood,
            self._rule_connection_errors,
            self._rule_suspicious_service,
            self._rule_land_attack,
        ]
        self.rule_names = [
            "SYN Flood (DoS)", "Port Scan (Probe)", "Brute Force (R2L)",
            "ICMP/Smurf Flood (DoS)", "Taux d'erreurs anormal",
            "Service suspect", "Land Attack",
        ]

    # ---------- Règles de signatures (seuils fixés par des experts) ----------
    def _rule_syn_flood(self, row):
        """Signature DoS classique : beaucoup de connexions + erreurs SYN."""
        return row['count'] > 100 and row['serror_rate'] > 0.7

    def _rule_port_scan(self, row):
        """Signature de scan : beaucoup d'hôtes/services différents."""
        return row['dst_host_count'] > 100 and row['dst_host_diff_srv_rate'] > 0.5

    def _rule_brute_force(self, row):
        """Signature brute-force : tentatives de connexion échouées."""
        return row['num_failed_logins'] >= 3

    def _rule_icmp_flood(self, row):
        """Signature Smurf/ICMP flood."""
        return row['protocol_type'] == 'icmp' and row['src_bytes'] > 800

    def _rule_connection_errors(self, row):
        """Taux de rejets anormalement élevé."""
        return row['rerror_rate'] > 0.8 or row['srv_rerror_rate'] > 0.8

    def _rule_suspicious_service(self, row):
        """Services historiquement ciblés avec flag anormal."""
        return row['service'] in ('private', 'eco_i') and row['flag'] in ('S0', 'REJ')

    def _rule_land_attack(self, row):
        """Land attack : paquet avec source = destination."""
        return row['land'] == 1

    def predict_one(self, row):
        """Retourne (is_attack, regle_declenchee)."""
        for rule, name in zip(self.rules, self.rule_names):
            try:
                if rule(row):
                    return 1, name
            except (KeyError, TypeError):
                continue
        return 0, None

    def predict(self, df):
        """Prédit sur un DataFrame entier."""
        preds = []
        triggered = []
        for _, row in df.iterrows():
            p, rule = self.predict_one(row)
            preds.append(p)
            triggered.append(rule)
        return np.array(preds), triggered


if __name__ == "__main__":
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, confusion_matrix)
    import json

    print("=" * 65)
    print("DÉTECTEUR CLASSIQUE À SIGNATURES (type Snort/Suricata)")
    print("=" * 65)

    df_test = load_nsl_kdd('test')
    ids = RuleBasedIDS()

    print(f"\nAnalyse de {len(df_test)} connexions avec {len(ids.rules)} règles fixes...")
    y_pred, triggered = ids.predict(df_test)
    y_true = df_test['binary_label'].values

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nRésultats du détecteur CLASSIQUE :")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}  <- rate beaucoup d'attaques")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print(f"\nMatrice de confusion :")
    print(f"                 Prédit Normal | Prédit Attaque")
    print(f"  Réel Normal  : {cm[0][0]:>12} | {cm[0][1]:>13}")
    print(f"  Réel Attaque : {cm[1][0]:>12} | {cm[1][1]:>13}")

    # Sauvegarder pour la comparaison
    RESULTS = Path(__file__).parent.parent / "results"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "rule_based_metrics.json").write_text(
        json.dumps({'metrics': metrics, 'confusion_matrix': cm.tolist()},
                   indent=2, ensure_ascii=False))
    print(f"\n  Résultats sauvegardés pour la comparaison IA vs Classique.")
