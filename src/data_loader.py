"""
Chargement et préparation du dataset réel NSL-KDD
===================================================
NSL-KDD est le dataset de référence pour la recherche en détection d'intrusion.
Il contient des connexions réseau réelles labellisées : trafic normal + 4 grandes
familles d'attaques (DoS, Probe, R2L, U2R) réparties en 39 types d'attaques.

Source : https://www.unb.ca/cic/datasets/nsl.html
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Les 41 colonnes du dataset NSL-KDD + label + difficulté
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Mapping des 39 types d'attaques vers les 4 grandes catégories
ATTACK_MAP = {
    # DoS - Denial of Service
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
    'udpstorm': 'DoS', 'worm': 'DoS',
    # Probe - Surveillance / Scan
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L - Remote to Local
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'httptunnel': 'R2L',
    # U2R - User to Root
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
    # Normal
    'normal': 'Normal',
}


def load_nsl_kdd(which='train'):
    """Charge le dataset NSL-KDD (train ou test)."""
    fname = 'KDDTrain.txt' if which == 'train' else 'KDDTest.txt'
    path = DATA_DIR / fname
    df = pd.read_csv(path, names=COLUMNS)

    # Supprimer la colonne difficulty (non pertinente)
    df = df.drop(columns=['difficulty'])

    # Catégoriser les attaques
    df['attack_category'] = df['label'].map(ATTACK_MAP).fillna('Unknown')

    # Label binaire : 0 = normal, 1 = attaque
    df['binary_label'] = (df['attack_category'] != 'Normal').astype(int)

    return df


def get_dataset_summary(df):
    """Retourne un résumé du dataset."""
    summary = {
        'total': len(df),
        'normal': int((df['attack_category'] == 'Normal').sum()),
        'attacks': int((df['attack_category'] != 'Normal').sum()),
        'by_category': df['attack_category'].value_counts().to_dict(),
        'n_attack_types': df['label'].nunique(),
    }
    return summary


if __name__ == "__main__":
    print("=" * 65)
    print("CHARGEMENT DU DATASET RÉEL NSL-KDD")
    print("=" * 65)

    for which in ['train', 'test']:
        df = load_nsl_kdd(which)
        s = get_dataset_summary(df)
        print(f"\n[{which.upper()}]")
        print(f"  Total           : {s['total']:>7} connexions réelles")
        print(f"  Trafic normal   : {s['normal']:>7} ({s['normal']/s['total']:.1%})")
        print(f"  Attaques        : {s['attacks']:>7} ({s['attacks']/s['total']:.1%})")
        print(f"  Types d'attaques: {s['n_attack_types']} distincts")
        print(f"  Par catégorie   :")
        for cat, n in sorted(s['by_category'].items(), key=lambda x: -x[1]):
            print(f"      {cat:<10}: {n:>7} ({n/s['total']:.1%})")
