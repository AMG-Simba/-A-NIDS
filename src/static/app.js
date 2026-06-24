let state = { total: 0, aiDetect: 0, ruleDetect: 0, blocked: 0, rows: [] };

async function loadInfo() {
  const r = await fetch('/api/info');
  const d = await r.json();
  document.getElementById('model-name').textContent = d.model_name;
  document.getElementById('threshold').textContent = d.threshold;
}

function levelTag(lvl) {
  const c = { CRITICAL:'t-crit', HIGH:'t-high', MEDIUM:'t-med', LOW:'t-low', INFO:'t-info' }[lvl] || 't-info';
  return `<span class="tag ${c}">${lvl}</span>`;
}

function render() {
  const tb = document.getElementById('rows');
  if (!state.rows.length) {
    tb.innerHTML = '<tr class="empty"><td colspan="7">Aucune analyse</td></tr>';
    return;
  }
  tb.innerHTML = state.rows.slice(0, 60).map(r => {
    // Verdict : qui a raison ?
    let verdict;
    if (r.is_real_attack) {
      if (r.ai_attack && !r.rule_attack) verdict = '<span class="verdict-good">IA rattrape ✓</span>';
      else if (r.ai_attack && r.rule_attack) verdict = '<span class="verdict-good">Les 2 ✓</span>';
      else if (!r.ai_attack && r.rule_attack) verdict = '<span class="verdict-miss">Classique seul</span>';
      else verdict = '<span class="verdict-bad">Les 2 ratent ✗</span>';
    } else {
      verdict = (r.ai_attack || r.rule_attack)
        ? '<span class="verdict-miss">Faux positif</span>'
        : '<span class="verdict-good">OK normal</span>';
    }
    const trueTag = r.is_real_attack
      ? `<span class="tag t-attack">${r.true_label}</span>`
      : `<span class="tag t-normal">normal</span>`;
    return `<tr>
      <td><code>${r.ip}</code></td>
      <td>${trueTag}</td>
      <td>${r.ai_attack ? '⚠ Attaque' : '✓ Normal'}</td>
      <td>${(r.ai_proba*100).toFixed(0)}%</td>
      <td>${levelTag(r.ai_level)}</td>
      <td>${r.rule_attack ? '⚠ '+(r.rule_name||'Attaque') : '✓ Normal'}</td>
      <td>${verdict}</td>
    </tr>`;
  }).join('');
}

function updateKPIs() {
  document.getElementById('k-total').textContent = state.total;
  document.getElementById('k-ai-detect').textContent = state.aiDetect;
  document.getElementById('k-rule-detect').textContent = state.ruleDetect;
  document.getElementById('k-blocked').textContent = state.blocked;
  document.getElementById('gain-ai').textContent = state.aiDetect;
  document.getElementById('gain-rule').textContent = state.ruleDetect;
  const diff = state.rows.filter(r => r.is_real_attack && r.ai_attack && !r.rule_attack).length;
  document.getElementById('gain-diff').textContent = diff;
}

async function simulate(n) {
  const r = await fetch('/api/simulate', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ n })
  });
  const d = await r.json();
  d.results.forEach(res => {
    state.rows.unshift(res);
    state.total++;
    if (res.ai_attack) state.aiDetect++;
    if (res.rule_attack) state.ruleDetect++;
    if (res.ai_action && res.ai_action.includes('bloquee')) state.blocked++;
  });
  render(); updateKPIs();
}

async function reset() {
  await fetch('/api/reset', { method: 'POST' });
  state = { total: 0, aiDetect: 0, ruleDetect: 0, blocked: 0, rows: [] };
  render(); updateKPIs();
}

loadInfo();
