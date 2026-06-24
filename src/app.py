"""
Dashboard web A-NIDS — Flask
=============================
Interface de supervision de l'agent IA, avec comparaison en direct
IA vs détecteur classique sur de vraies connexions NSL-KDD.
"""

from pathlib import Path
import json

from flask import Flask, render_template, jsonify, request

from data_loader import load_nsl_kdd
from ai_agent import AIAgent
from rule_based_ids import RuleBasedIDS

BASE = Path(__file__).parent.parent
RESULTS = BASE / "results"

app = Flask(__name__,
            template_folder=str(BASE / "src" / "templates"),
            static_folder=str(BASE / "src" / "static"))

agent = AIAgent()
rule_ids = RuleBasedIDS()
df_test = load_nsl_kdd('test')


@app.route('/')
def home():
    return render_template('dashboard.html')


@app.route('/api/info')
def info():
    """Infos sur l'agent et les performances chargées."""
    data = {'model_name': agent.model_name, 'threshold': agent.threshold}
    cmp_file = RESULTS / "comparison.json"
    if cmp_file.exists():
        data['comparison'] = json.loads(cmp_file.read_text())
    return jsonify(data)


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Tire N connexions réelles du jeu de test et les fait analyser par
    l'agent IA ET le détecteur classique, pour comparer en direct."""
    n = min(int(request.json.get('n', 10)), 50)
    sample = df_test.sample(n).reset_index(drop=True)

    results = []
    for i, row in sample.iterrows():
        traffic = row.drop(['label', 'attack_category', 'binary_label'],
                           errors='ignore').to_dict()
        ip = f"10.0.{i}.{(i * 7) % 254 + 1}"

        # Agent IA
        ai_rep = agent.analyze(traffic, source_ip=ip)
        # Détecteur classique
        rule_pred, rule_name = rule_ids.predict_one(row)

        results.append({
            'ip': ip,
            'true_label': row['attack_category'],
            'is_real_attack': bool(row['binary_label']),
            'ai_attack': ai_rep['is_attack'],
            'ai_proba': ai_rep['attack_probability'],
            'ai_level': ai_rep['threat_level'],
            'ai_profile': ai_rep['profile'],
            'ai_action': ai_rep['action_taken'],
            'rule_attack': bool(rule_pred),
            'rule_name': rule_name,
        })

    return jsonify({'results': results})


@app.route('/api/report')
def report():
    return jsonify(agent.network_report())


@app.route('/api/reset', methods=['POST'])
def reset():
    agent.alerts.clear()
    agent.blocked_ips.clear()
    return jsonify({'ok': True})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Dashboard A-NIDS — http://localhost:5000")
    print(f"  Agent IA : {agent.model_name} (seuil {agent.threshold})")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
