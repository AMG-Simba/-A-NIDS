"""
Agent IA — A-NIDS  (détecteur opérationnel)
============================================
L'agent IA au cœur du système. Il analyse les flux réseau en temps réel,
classifie (normal/attaque), identifie le profil, évalue la menace sur
5 niveaux, alerte et bloque automatiquement les IP malveillantes.

C'est l'élément central du projet — un vrai IDS piloté par IA.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
MODELS = BASE / "models"
RESULTS = BASE / "results"
LOGS = RESULTS / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(LOGS / "agent.log", encoding='utf-8'),
              logging.StreamHandler()]
)
logger = logging.getLogger("A-NIDS")

CATEGORICAL = ['protocol_type', 'service', 'flag']


class AIAgent:
    """Agent IA de détection d'intrusion."""

    THREAT_LEVELS = [
        (0.00, 0.50, "INFO", "Trafic normal"),
        (0.50, 0.70, "LOW", "Activite suspecte legere"),
        (0.70, 0.85, "MEDIUM", "Activite suspecte"),
        (0.85, 0.95, "HIGH", "Attaque probable"),
        (0.95, 1.01, "CRITICAL", "Attaque confirmee - blocage immediat"),
    ]

    def __init__(self):
        logger.info("Initialisation de l'agent IA...")
        self.model = joblib.load(MODELS / "best_model.pkl")
        self.scaler = joblib.load(MODELS / "scaler.pkl")
        self.encoders = joblib.load(MODELS / "encoders.pkl")
        self.feature_names = joblib.load(MODELS / "feature_names.pkl")
        self.model_name = (MODELS / "best_model_name.txt").read_text().strip()
        # Seuil de décision optimisé (calculé par optimize_agent.py)
        thresh_file = MODELS / "optimal_threshold.txt"
        self.threshold = float(thresh_file.read_text().strip()) if thresh_file.exists() else 0.50
        self.blocked_ips = set()
        self.whitelist = {"192.168.1.1", "10.0.0.1", "8.8.8.8"}  # IPs de confiance
        self.alerts = []
        logger.info(f"Agent IA pret - modele : {self.model_name}")

    def preprocess(self, traffic):
        df = pd.DataFrame([traffic]) if isinstance(traffic, dict) else traffic.copy()
        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
        # Compléter les features manquantes avec 0
        for f in self.feature_names:
            if f not in df.columns:
                df[f] = 0
        return self.scaler.transform(df[self.feature_names].values)

    def predict(self, traffic):
        X = self.preprocess(traffic)
        proba = float(self.model.predict_proba(X)[0][1])
        # Décision basée sur le seuil optimisé (et non le 0.5 par défaut)
        pred = int(proba >= self.threshold)
        return pred, proba

    def threat_level(self, proba):
        for lo, hi, level, msg in self.THREAT_LEVELS:
            if lo <= proba < hi:
                return level, msg
        return "INFO", "Trafic normal"

    def identify_profile(self, traffic):
        """Identifie le type d'attaque par règles métier (features NSL-KDD réelles)."""
        count = traffic.get('count', 0)
        serror = traffic.get('serror_rate', 0)
        diff_srv = traffic.get('dst_host_diff_srv_rate', 0)
        failed = traffic.get('num_failed_logins', 0)
        dst_host = traffic.get('dst_host_count', 0)
        duration = traffic.get('duration', 0)
        logged = traffic.get('logged_in', 0)
        root_shell = traffic.get('root_shell', 0)
        num_root = traffic.get('num_root', 0)
        guest = traffic.get('is_guest_login', 0)
        hot = traffic.get('hot', 0)
        service = traffic.get('service', '')
        srv_count = traffic.get('srv_count', 0)

        # U2R : élévation de privilèges (root shell, accès root)
        if root_shell == 1 or num_root > 0 or traffic.get('num_file_creations', 0) > 1:
            return "U2R / Elevation de privileges"
        # R2L : accès distant (login invité, tentatives échouées, hot indicators)
        if failed >= 2 or guest == 1 or hot > 2:
            return "R2L / Acces distant non autorise"
        # DoS : flood (beaucoup de connexions + erreurs)
        if count > 100 and (serror > 0.3 or count > 400):
            return "DoS / DDoS (flood)"
        # Probe : scan (nombreux hôtes/services différents)
        if dst_host > 100 and diff_srv > 0.3:
            return "Probe / Scan de ports"
        if srv_count > 50 or count > 150:
            return "DoS / DDoS (flood)"
        return "Attaque generique detectee"

    def block_ip(self, ip, reason):
        if ip in self.whitelist:
            logger.info(f"IP {ip} en whitelist - blocage ignore")
            return False
        if ip in self.blocked_ips:
            return False
        self.blocked_ips.add(ip)
        logger.warning(f"IP BLOQUEE : {ip} ({reason})")
        with open(LOGS / "blocked_ips.log", "a", encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} | {ip} | {reason}\n")
        # En production : os.system(f"iptables -A INPUT -s {ip} -j DROP")
        return True

    def analyze(self, traffic, source_ip=None):
        if source_ip is None:
            source_ip = f"192.168.{random.randint(1,254)}.{random.randint(1,254)}"

        pred, proba = self.predict(traffic)
        level, msg = self.threat_level(proba)

        report = {
            'source_ip': source_ip,
            'is_attack': bool(pred),
            'attack_probability': round(proba, 4),
            'threat_level': level,
            'message': msg,
            'profile': None,
            'action_taken': 'aucune',
            'timestamp': datetime.now().isoformat(),
        }

        if pred == 1 or proba > 0.5:
            report['profile'] = self.identify_profile(traffic)
            self.alerts.append(report)
            if level in ("HIGH", "CRITICAL"):
                blocked = self.block_ip(source_ip, report['profile'])
                report['action_taken'] = 'IP bloquee' if blocked else 'deja bloquee/whitelist'
            else:
                report['action_taken'] = 'Alerte generee'

        return report

    def network_report(self):
        if not self.alerts:
            return {'status': 'Aucune alerte', 'total_alerts': 0}
        df = pd.DataFrame(self.alerts)
        return {
            'timestamp': datetime.now().isoformat(),
            'total_alerts': len(self.alerts),
            'threats': df['threat_level'].value_counts().to_dict(),
            'profiles': df['profile'].value_counts().to_dict(),
            'blocked_ips': sorted(self.blocked_ips),
            'n_blocked': len(self.blocked_ips),
        }


def run_demo():
    """Démonstration sur de vraies connexions du dataset NSL-KDD."""
    from data_loader import load_nsl_kdd

    print("=" * 65)
    print("DÉMONSTRATION DE L'AGENT IA — A-NIDS (données réelles)")
    print("=" * 65)

    agent = AIAgent()
    df = load_nsl_kdd('test')

    # Échantillon mixte réel
    samples = pd.concat([
        df[df['attack_category'] == 'Normal'].sample(4, random_state=1),
        df[df['attack_category'] == 'DoS'].sample(3, random_state=1),
        df[df['attack_category'] == 'Probe'].sample(2, random_state=1),
        df[df['attack_category'] == 'R2L'].sample(2, random_state=1),
        df[df['attack_category'] == 'U2R'].sample(1, random_state=1),
    ]).sample(frac=1, random_state=7).reset_index(drop=True)

    print(f"\nAnalyse de {len(samples)} connexions réelles en temps réel...\n")
    for i, row in samples.iterrows():
        traffic = row.drop(['label', 'attack_category', 'binary_label']).to_dict()
        true_cat = row['attack_category']
        ip = f"10.0.0.{i+20}"
        rep = agent.analyze(traffic, ip)
        icon = "[ATTAQUE]" if rep['is_attack'] else "[NORMAL] "
        print(f"{icon} #{i+1:>2} IP={ip:<12} reel={true_cat:<7} "
              f"P={rep['attack_probability']:.3f} {rep['threat_level']:<9} "
              f"-> {rep['action_taken']}")
        if rep['profile']:
            print(f"            Profil identifie : {rep['profile']}")

    print("\n" + "=" * 65)
    print("RAPPORT RÉSEAU")
    print("=" * 65)
    print(json.dumps(agent.network_report(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_demo()
