"""
A-NIDS — Pipeline complet
==========================
Lance toutes les étapes du projet dans l'ordre :
  1. Vérification du dataset réel NSL-KDD
  2. Détecteur classique à signatures
  3. Entraînement de l'agent IA (6 modèles)
  4. Optimisation du seuil de décision
  5. Comparaison IA vs Classique + graphiques
  6. Démo de l'agent IA

Usage : python run_all.py
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent

STEPS = [
    ("data_loader.py",    "Vérification du dataset réel NSL-KDD"),
    ("rule_based_ids.py", "Évaluation du détecteur CLASSIQUE à signatures"),
    ("train_agent.py",    "Entraînement de l'agent IA (6 modèles)"),
    ("optimize_agent.py", "Optimisation du seuil de décision"),
    ("compare.py",        "Comparaison IA vs Classique + graphiques"),
    ("ai_agent.py",       "Démonstration de l'agent IA"),
]


def run():
    print("\n" + "=" * 70)
    print("  A-NIDS — LANCEMENT DU PIPELINE COMPLET")
    print("=" * 70)

    for i, (script, desc) in enumerate(STEPS, 1):
        print(f"\n\n{'#' * 70}")
        print(f"#  ÉTAPE {i}/{len(STEPS)} : {desc}")
        print(f"{'#' * 70}\n")
        result = subprocess.run([sys.executable, str(SRC / script)])
        if result.returncode != 0:
            print(f"\n[!] Erreur à l'étape {i} ({script}). Arrêt.")
            return
    print("\n\n" + "=" * 70)
    print("  PIPELINE TERMINÉ — voir le dossier results/ pour les graphiques")
    print("=" * 70)


if __name__ == "__main__":
    run()
