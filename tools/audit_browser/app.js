/* Selection Audit Browser: vanilla JS, reads window.AUDIT from data.js. */
(function () {
  'use strict';
  const A = window.AUDIT;
  const C = { qwen: '#3B8ED0', smol: '#E4572E', random: '#9AA0A6', good: '#2EAD5B', bad: '#E04848', agree: '#8E6CC7', ink: '#1F2A33', ink3: '#737B83', rule: '#DDDDDD', goodText: '#23884A', badText: '#C63A3A', agreeText: '#7556AE' };
  const app = document.getElementById('app');
  const overlay = document.getElementById('overlay');
  const S = A.summary;
  const P = A.prompts;
  const byId = new Map(P.map(p => [p.id, p]));
  const JN = { qwen: 'Qwen', smol: 'Smol' };
  document.getElementById('mnote').textContent = `${A.meta.n_prompts} prompts · ${new Set(A.prompts.map(p => p.country)).size} countries · 2 VLM judges × 3 presentation orders`;

  // ---------- helpers ----------
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const f3 = x => (x == null ? '–' : (Math.abs(x) < 5e-4 ? 0 : x).toFixed(3));
  const sg = x => (x == null ? '–' : (x > 5e-4 ? '+' : x < -5e-4 ? '−' : '±') + Math.abs(x).toFixed(3));
  const pct = x => (x == null ? '–' : (100 * x).toFixed(1) + '%');
  const ci = c => (c && c[0] != null ? `${sg(c[0])} to ${sg(c[1])}` : '');
  const svgEl = (w, h, inner) => `<svg viewBox="0 0 ${w} ${h}" width="100%" preserveAspectRatio="xMidYMid meet" role="img" style="display:block">${inner}</svg>`;
  const t = (x, y, s, o = {}) => `<text x="${x}" y="${y}" font-size="${o.size || 12}" fill="${o.fill || C.ink}" text-anchor="${o.anchor || 'start'}" font-weight="${o.weight || 400}" font-family="${o.serif ? 'Georgia, serif' : '-apple-system, Helvetica Neue, Arial, sans-serif'}" ${o.extra || ''}>${s}</text>`;
  const utilColor = u => (u >= 0.75 ? C.good : u >= 0.5 ? '#D9A21B' : C.bad);
  const signCls = v => (v > 5e-4 ? 'good' : v < -5e-4 ? 'bad' : '');
  const regretCls = r => (r < 1e-9 ? 'good' : r > 0.25 ? 'bad' : '');
  const retry = `onerror="if(!this.dataset.r){this.dataset.r=1;this.src=this.src+'?r=1'}"`;
  const GEN = { SD35: 'SD-3.5', flux: 'Flux', 'gpt-image': 'GPT-Image', imagegen3: 'Imagen 3' };
  const GS = { 'gpt-image': 'GPT-Img', imagegen3: 'Imagen3' };
  const gen = c => GEN[c.gen] || c.gen;
  const tags = (q, s) => `<div class="tags">${q ? '<span class="tag q">Q</span>' : ''}${s ? '<span class="tag s">S</span>' : ''}</div>`;

  // ---------- routing ----------
  const blank = () => ({ country: '', cat: '', outcome: '', judge: 'qwen', q: '', sort: 'order' });
  const state = { view: 'overview', f: blank(), detail: null, dj: 'both' };
  const readF = params => ({ country: params.get('country') || '', cat: params.get('category') || '', outcome: params.get('outcome') || '', judge: params.get('judge') || 'qwen', q: params.get('q') || '', sort: params.get('sort') || 'order' });
  function parseHash() {
    const h = decodeURIComponent(location.hash.replace(/^#/, ''));
    const [head, qs] = h.split('?');
    const params = new URLSearchParams(qs || '');
    if (head.startsWith('detail=')) {
      const parts = new URLSearchParams(head);
      const key = parts.get('detail') || '';
      const p = P.find(x => x.id.startsWith(key));
      state.detail = p ? p.id : null;
      state.dj = parts.get('judge') || 'both';
      state.view = 'explore';
      if (qs) state.f = readF(params);
    } else {
      state.detail = null;
      state.view = head === 'explore' ? 'explore' : 'overview';
      if (state.view === 'explore') state.f = readF(params);
    }
    render();
  }
  function exploreHash() {
    const f = state.f, p = new URLSearchParams();
    if (f.country) p.set('country', f.country);
    if (f.cat) p.set('category', f.cat);
    if (f.outcome) p.set('outcome', f.outcome);
    if (f.judge !== 'qwen') p.set('judge', f.judge);
    if (f.q) p.set('q', f.q);
    if (f.sort !== 'order') p.set('sort', f.sort);
    const s = p.toString();
    return '#explore' + (s ? '?' + s : '');
  }
  function setFilter(k, v) {
    state.f[k] = v;
    history.replaceState(null, '', exploreHash());
    renderExplore(true);
  }
  function render() {
    document.querySelectorAll('.nav a').forEach(el => el.classList.toggle('active', el.dataset.view === state.view));
    if (state.view === 'overview') renderOverview(); else renderExplore();
    renderDetail();
  }

  // ---------- overview ----------
  const fig = (what, n, unit, cap) => `<div class="figure"><div class="what">${what}</div><div class="n">${n}${unit ? `<small>${unit}</small>` : ''}</div><div class="cap">${cap}</div></div>`;

  function renderOverview() {
    const q = S['Qwen'], r = S['Random'], un = S['Qwen unanimous'], cm = S['Cross-model'];
    const qa = A.position.qwen.slots.A;
    const nP = A.meta.n_prompts;
    const figs = [
      fig('Qwen regret', f3(q.regret), '', `vs. <b>${f3(r.regret)}</b> for a random pick`),
      fig('Gain over random', sg(q.gain), '', `95% CI ${ci(q.gain_ci)}`),
      fig('Below-mean picks', pct(q.hsr), '', `vs. <b>${pct(q.random_hsr)}</b> for a random pick`),
      fig('Qwen picks in slot A', pct(qa.rate), '', `vs. <b>${pct(qa.uniform_rate)}</b> if slot-blind`),
      fig('Unanimity gate keeps', pct(un.coverage), `${un.n} of ${nP}`, `kept gain <b class="${signCls(un.gain)}">${sg(un.gain)}</b>`),
      fig('Cross-model gate keeps', pct(cm.coverage), `${cm.n} of ${nP}`, `kept gain <b class="${signCls(cm.gain)}">${sg(cm.gain)}</b>`),
    ].join('');
    app.innerHTML = `<div class="page">
      <div class="lede"><h1>Audit overview</h1>
        <p>Each prompt has 3–4 candidate images; utility is the mean human alignment rating (0–1). Judges pick one image per presentation order.</p></div>
      <div class="figures">${figs}</div>
      <div class="cols a">
        <section class="sec"><h2>Judges favour early slots</h2><p class="dek">Share of valid picks by presentation slot. Dashes mark the rate expected from a slot-blind judge.</p>
          ${slotChart()}</section>
        <section class="sec"><h2>Regret by country</h2><p class="dek">Mean regret (best utility − chosen utility) on each country's 30 prompts, original order. Lower is better.</p>
          ${countryChart()}</section>
      </div>
      <div class="cols b rule-top">
        <section class="sec"><h2>What agreement gates keep, and what they throw away</h2><p class="dek">Qwen's original-order pick, scored as gain over random, on the prompts each gate keeps and on those it rejects. Bars show 95% intervals.</p>
          ${gateTable()}</section>
        <section class="sec"><h2>Gain over random, by policy</h2><p class="dek">Mean utility gain on the prompts each policy answers; right-hand column is the share answered.</p>
          ${forestChart()}</section>
      </div></div>`;
  }

  function slotChart() {
    const W = 640, H = 232, m = { l: 36, r: 8, t: 22, b: 26 };
    const slots = ['A', 'B', 'C', 'D'];
    const pq = A.position.qwen.slots, ps = A.position.smol.slots;
    const top = Math.ceil(Math.max(...slots.map(s => Math.max(pq[s].rate, ps[s].rate))) * 10) / 10;
    const y = v => m.t + (H - m.t - m.b) * (1 - v / top);
    const gw = (W - m.l - m.r) / 4, bw = gw * 0.28;
    let g = '', labs = '';
    for (let v = 0.1; v <= top + 1e-9; v += 0.1) {
      g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="#EDEDED"/>`;
      g += t(m.l - 6, y(v) + 4, Math.round(v * 100) + '%', { anchor: 'end', size: 11.5, fill: C.ink3 });
    }
    slots.forEach((s, i) => {
      const cx = m.l + gw * i + gw / 2;
      [[pq[s].rate, C.qwen, -1, 'Qwen'], [ps[s].rate, C.smol, 1, 'Smol']].forEach(([v, col, side, name]) => {
        const x = cx + (side < 0 ? -bw - 1 : 1);
        g += `<rect x="${x}" y="${y(v)}" width="${bw}" height="${y(0) - y(v)}" fill="${col}"/>`;
        labs += t(x + bw / 2, y(v) - 5, Math.round(v * 100) + '%', { anchor: 'middle', size: 12, fill: C.ink, extra: 'stroke="#fff" stroke-width="4" paint-order="stroke"' });
        if (i === 0) g += t(x + bw / 2, y(v) - 20, name, { anchor: 'middle', size: 12, weight: 700, fill: col });
      });
      const u = pq[s].uniform_rate;
      g += `<line x1="${cx - bw - 9}" x2="${cx + bw + 9}" y1="${y(u)}" y2="${y(u)}" stroke="${C.ink}" stroke-width="1.5" stroke-dasharray="4 3"/>`;
      if (i === 2) g += t(cx + bw + 12, y(u) + 4, 'slot-blind rate', { size: 11.5, fill: C.ink3 });
      g += t(cx, H - 7, s, { anchor: 'middle', size: 13, weight: 700 });
    });
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(0)}" y2="${y(0)}" stroke="${C.ink}"/>` + labs;
    const foot = `<p class="note">Slot shown to the judge. Qwen: ${A.position.qwen.calls} valid calls, χ²(3) = ${A.position.qwen.chi2.toFixed(1)}; Smol: ${A.position.smol.calls} valid calls, χ²(3) = ${A.position.smol.chi2.toFixed(1)}.</p>`;
    return svgEl(W, H, g) + foot;
  }

  function countryChart() {
    const rows = {};
    A.country.forEach(r => { (rows[r.country] = rows[r.country] || {})[r.policy] = r; });
    const names = Object.keys(rows).sort((a, b) => rows[a]['Qwen'].regret - rows[b]['Qwen'].regret);
    const W = 860, rh = 22, m = { l: 96, r: 16, t: 24, b: 22 }, H = m.t + m.b + rh * names.length;
    const top = Math.ceil(Math.max(...A.country.filter(r => r.policy !== 'Qwen unanimous').map(r => r.regret)) * 10) / 10;
    const x = v => m.l + (W - m.l - m.r) * v / top;
    let g = '';
    for (let v = 0; v <= top + 1e-9; v += 0.1) {
      g += `<line x1="${x(v)}" x2="${x(v)}" y1="${m.t - 4}" y2="${H - m.b}" stroke="#EDEDED"/>`;
      g += t(x(v), H - 6, v.toFixed(1), { anchor: 'middle', size: 11.5, fill: C.ink3 });
    }
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="${C.ink}"/>`;
    names.forEach((n, i) => {
      const cy = m.t + rh * i + rh / 2, R = rows[n];
      const vals = [R['Random'].regret, R['Qwen'].regret, R['Smol'].regret];
      g += t(m.l - 12, cy + 4, esc(n), { anchor: 'end', size: 13 });
      g += `<line x1="${x(Math.min(...vals))}" x2="${x(Math.max(...vals))}" y1="${cy}" y2="${cy}" stroke="#CFCFCF" stroke-width="1"/>`;
      [['Random', C.random], ['Smol', C.smol], ['Qwen', C.qwen]].forEach(([k, col]) => {
        g += `<circle cx="${x(R[k].regret)}" cy="${cy}" r="5" fill="${col}"/>`;
        if (i === 0) g += t(x(R[k].regret), cy - 11, k, { anchor: 'middle', size: 12, weight: 700, fill: k === 'Random' ? '#6F757B' : col });
      });
    });
    return svgEl(W, H, g);
  }

  function ciGlyph(est, c, lo, hi, col) {
    const W = 130, H = 18, x = v => 4 + (W - 8) * (v - lo) / (hi - lo);
    let g = `<line x1="${x(0)}" x2="${x(0)}" y1="0" y2="${H}" stroke="#BDBDBD"/>`;
    g += `<line x1="${x(c[0])}" x2="${x(c[1])}" y1="${H / 2}" y2="${H / 2}" stroke="${col}" stroke-width="1.5"/>`;
    g += `<rect x="${x(est) - 3.5}" y="${H / 2 - 3.5}" width="7" height="7" fill="${col}"/>`;
    return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="display:block">${g}</svg>`;
  }

  function gateTable() {
    const rows = [
      ['No gate', "Qwen's original-order pick", S['Qwen'], null],
      ['Unanimity', 'Same Qwen pick in all three orders', S['Qwen unanimous'], S['Qwen, not unanimous']],
      ['Cross-model', 'Qwen and Smol picks match', S['Cross-model'], S['Qwen, judges disagree']],
    ];
    const all = rows.flatMap(r => [r[2], r[3]]).filter(Boolean);
    const lo = Math.min(0, ...all.map(s => s.gain_ci[0])) - 0.01, hi = Math.max(0, ...all.map(s => s.gain_ci[1])) + 0.01;
    const cell = (s, col) => s ? `<td><div style="display:flex;align-items:center;gap:12px">${ciGlyph(s.gain, s.gain_ci, lo, hi, col)}
        <div style="white-space:nowrap"><b class="num ${signCls(s.gain)}">${sg(s.gain)}</b> <span class="muted num" style="font-size:12.5px">n = ${s.n}</span></div></div></td>`
      : '<td class="muted" style="font-size:13px">none rejected</td>';
    return `<table class="t"><thead><tr><th>Gate</th><th>Kept prompts</th><th>Rejected prompts</th><th class="r">Kept regret</th></tr></thead><tbody>
      ${rows.map(r => `<tr><td class="gname">${r[0]}<small>${r[1]}</small></td>${cell(r[2], r[0] === 'No gate' ? C.qwen : C.agree)}${cell(r[3], C.random)}<td class="r num">${f3(r[2].regret)}</td></tr>`).join('')}
    </tbody></table>
    <p class="note">A useful gate should keep prompts where the judge does better than on the prompts it rejects. Unanimity does (${sg(S['Qwen unanimous'].gain)} vs. ${sg(S['Qwen, not unanimous'].gain)}); cross-model agreement does the reverse (${sg(S['Cross-model'].gain)} vs. ${sg(S['Qwen, judges disagree'].gain)}).</p>`;
  }

  function forestChart() {
    const pols = [['Qwen', C.qwen], ['Qwen majority', C.qwen], ['Qwen agree 2/3', C.qwen], ['Qwen unanimous', C.agree], ['Smol', C.smol], ['Smol majority', C.smol], ['Smol agree 2/3', C.smol], ['Cross-model', C.agree]].filter(p => S[p[0]]);
    const W = 640, rh = 25, m = { l: 122, r: 128, t: 22, b: 24 }, H = m.t + m.b + rh * pols.length;
    const lo = Math.floor(Math.min(...pols.map(p => S[p[0]].gain_ci[0])) * 20) / 20, hi = Math.ceil(Math.max(...pols.map(p => S[p[0]].gain_ci[1])) * 20) / 20;
    const x = v => m.l + (W - m.l - m.r) * (v - lo) / (hi - lo);
    let g = '';
    for (let v = lo; v <= hi + 1e-9; v += 0.05) {
      const z = Math.abs(v) < 1e-9;
      g += `<line x1="${x(v)}" x2="${x(v)}" y1="${m.t - 6}" y2="${H - m.b}" stroke="${z ? '#8C8C8C' : '#EDEDED'}"/>`;
      g += t(x(v), H - 6, z ? '0' : (v > 0 ? '+' : '−') + Math.abs(v).toFixed(2), { anchor: 'middle', size: 11.5, fill: C.ink3 });
    }
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="${C.ink}"/>`;
    g += t(x(0) - 6, m.t - 10, '← worse than random', { anchor: 'end', size: 11, fill: C.ink3 }) + t(x(0) + 6, m.t - 10, 'better →', { size: 11, fill: C.ink3 });
    g += t(W - m.r + 14, m.t - 10, 'gain', { size: 11, fill: C.ink3 }) + t(W, m.t - 10, 'answered', { size: 11, fill: C.ink3, anchor: 'end' });
    pols.forEach(([k, col], i) => {
      const s = S[k], cy = m.t + rh * i + rh / 2;
      g += t(m.l - 12, cy + 4, k, { anchor: 'end', size: 12.5 });
      g += `<line x1="${x(s.gain_ci[0])}" x2="${x(s.gain_ci[1])}" y1="${cy}" y2="${cy}" stroke="${col}" stroke-width="1.5"/>`;
      g += `<rect x="${x(s.gain) - 4}" y="${cy - 4}" width="8" height="8" fill="${col}"/>`;
      g += t(W - m.r + 14, cy + 4, sg(s.gain), { size: 12.5, fill: C.ink });
      g += t(W, cy + 4, pct(s.coverage), { size: 12, fill: C.ink3, anchor: 'end' });
    });
    return svgEl(W, H, g);
  }

  // ---------- explorer ----------
  const OUTCOMES = [
    ['below', 'Pick below the mean', C.bad, (p, j) => p[j].m0 && p[j].m0.below_mean],
    ['best', 'Pick is human-best', C.good, (p, j) => p[j].m0 && p[j].m0.best],
    ['unanimous', 'Same pick in all orders', C.agree, (p, j) => p[j].unanimous],
    ['disagree', 'Orders disagree', '#D9A21B', (p, j) => p[j].orders_disagree],
    ['agree', 'Qwen and Smol agree', C.ink, p => p.judges_agree],
  ];
  function filtered(ignore) {
    const f = state.f, q = f.q.trim().toLowerCase();
    const out = OUTCOMES.find(o => o[0] === f.outcome);
    return P.filter(p => (ignore === 'country' || !f.country || p.country === f.country)
      && (ignore === 'cat' || !f.cat || p.category === f.cat)
      && (ignore === 'outcome' || !out || out[3](p, f.judge))
      && (!q || p.prompt.toLowerCase().includes(q) || p.id.startsWith(q)));
  }
  function sorted(list) {
    const j = state.f.judge, s = state.f.sort;
    const key = { regret: p => -(p[j].m0 ? p[j].m0.regret : -1), gain: p => (p[j].m0 ? p[j].m0.gain : 9), country: p => p.country + p.prompt }[s];
    if (!key) return list;
    return list.slice().sort((a, b) => { const ka = key(a), kb = key(b); return ka < kb ? -1 : ka > kb ? 1 : 0; });
  }
  function counts(list, fn) { const m = {}; list.forEach(p => { const k = fn(p); m[k] = (m[k] || 0) + 1; }); return m; }

  function renderExplore(keepFocus) {
    const f = state.f;
    const cC = counts(filtered('country'), p => p.country), cK = counts(filtered('cat'), p => p.category);
    const countries = [...new Set(P.map(p => p.country))].sort(), cats = [...new Set(P.map(p => p.category))].sort();
    const base = filtered('outcome');
    const list = sorted(filtered());
    const item = (k, v, label, n, sw) => `<li><button class="${f[k] === v ? 'on' : ''}" data-k="${k}" data-v="${esc(v)}"><span>${sw ? `<i class="sq" style="background:${sw}"></i>` : ''}<span class="nm">${esc(label)}</span></span>${n != null ? `<span class="c num">${n}</span>` : ''}</button></li>`;
    const side = `<aside class="side">
      <span class="label">Search</span><input class="search" id="q" placeholder="Prompt text or id" value="${esc(f.q)}">
      <span class="label">Country</span><ul class="flist two">${item('country', '', 'All', null)}${countries.map(c => item('country', c, c, cC[c] || 0)).join('')}</ul>
      <span class="label">Category</span><ul class="flist">${item('cat', '', 'All', null)}${cats.map(c => item('cat', c, c.replace(/-/g, ' '), cK[c] || 0)).join('')}</ul>
      <span class="label">Judge outcome</span>
      <div class="toggle">${['qwen', 'smol'].map(j => `<button class="${j} ${f.judge === j ? 'on' : ''}" data-judge="${j}">${JN[j]}</button>`).join('')}</div>
      <ul class="flist">${OUTCOMES.map(o => `<li><button class="${f.outcome === o[0] ? 'on' : ''}" data-outcome="${o[0]}"><span><i class="sq" style="background:${o[2]}"></i><span class="nm">${o[1]}</span></span><span class="c num">${base.filter(p => o[3](p, f.judge)).length}</span></button></li>`).join('')}</ul>
      <div class="side-foot">Outcomes refer to the selected judge's pick in the original order. Thumbnails show that order; <span class="tag q">Q</span> <span class="tag s">S</span> mark each judge's pick, ★ the human-best image, and the bar its alignment rating.<br><button class="linkbtn" id="reset">Clear filters</button></div>
    </aside>`;
    const desc = [f.country, f.cat && f.cat.replace(/-/g, ' '), f.outcome && `${JN[f.judge]}: ${OUTCOMES.find(o => o[0] === f.outcome)[1].toLowerCase()}`, f.q && `“${esc(f.q)}”`].filter(Boolean).join(' · ');
    const main = `<section>
      <div class="rhead"><div class="count num">${list.length} prompts<span>${desc || 'all countries and categories'}</span></div>
        <label class="sort">Sort by <select id="sort">${[['order', 'dataset order'], ['regret', `${JN[f.judge]} regret, high to low`], ['gain', `${JN[f.judge]} gain, low to high`], ['country', 'country']].map(([v, l]) => `<option value="${v}" ${f.sort === v ? 'selected' : ''}>${l}</option>`).join('')}</select></label></div>
      ${list.length ? `<div class="grid">${list.map((p, i) => entry(p, i)).join('')}</div>` : '<div class="empty">No prompts match these filters.</div>'}
    </section>`;
    app.innerHTML = `<div class="page"><div class="explore">${side}${main}</div></div>`;
    app.querySelectorAll('[data-k]').forEach(b => b.onclick = () => setFilter(b.dataset.k, b.dataset.v));
    app.querySelectorAll('[data-judge]').forEach(b => b.onclick = () => setFilter('judge', b.dataset.judge));
    app.querySelectorAll('[data-outcome]').forEach(b => b.onclick = () => setFilter('outcome', f.outcome === b.dataset.outcome ? '' : b.dataset.outcome));
    app.querySelector('#sort').onchange = e => setFilter('sort', e.target.value);
    app.querySelector('#reset').onclick = () => { state.f = blank(); history.replaceState(null, '', '#explore'); renderExplore(); };
    const qi = app.querySelector('#q');
    qi.oninput = e => setFilter('q', e.target.value);
    if (keepFocus === true && f.q) { qi.focus(); qi.setSelectionRange(qi.value.length, qi.value.length); }
    app.querySelectorAll('.entry').forEach(el => el.onclick = () => { location.hash = 'detail=' + el.dataset.id; });
  }

  function entry(p, idx) {
    const q0 = p.qwen.pick0, s0 = p.smol.pick0;
    const ord = p.qwen.orders[0] ? p.qwen.orders[0].order : p.candidates.map((_, i) => i);
    const thumbs = ord.map(i => {
      const c = p.candidates[i], best = p.best.includes(i);
      return `<div class="th ${best ? 'best' : ''}"><div class="img"><img ${idx >= 24 ? 'loading="lazy"' : ''} src="${c.thumb}" alt="" ${retry}>${tags(q0 === i, s0 === i)}</div>
        <div class="ubar"><i style="width:${100 * c.utility}%;background:${utilColor(c.utility)}"></i></div>
        <div class="cap"><span>${GS[c.gen] || gen(c)}</span><b class="num">${c.utility.toFixed(2)}</b></div></div>`;
    }).join('');
    const reg = j => { const m = p[j].m0; return m ? `<span>${JN[j]} regret <b class="num ${regretCls(m.regret)}">${f3(m.regret)}</b></span>` : `<span>${JN[j]}: <span class="bad">no valid pick</span></span>`; };
    const flag = [p.qwen.unanimous && 'Qwen unanimous', p.judges_agree && 'judges agree'].filter(Boolean).join(' · ');
    return `<article class="entry" data-id="${p.id}">
      <div class="meta">${esc(p.country)} · ${esc(p.category.replace(/-/g, ' '))}</div>
      <p class="ptext">${esc(p.prompt)}</p>
      <div class="thumbs" style="grid-template-columns:repeat(${Math.max(4, ord.length)},1fr)">${thumbs}</div>
      <div class="foot">${reg('qwen')}${reg('smol')}${flag ? `<span class="flag">${flag}</span>` : ''}</div>
    </article>`;
  }

  // ---------- detail ----------
  function renderDetail() {
    if (!state.detail) { overlay.innerHTML = ''; document.body.style.overflow = ''; return; }
    const p = byId.get(state.detail), dj = state.dj;
    document.body.style.overflow = 'hidden';
    const show = j => dj === 'both' || dj === j;
    const n = p.candidates.length;
    const labels = (p.qwen.orders[0] || p.smol.orders[0]).labels;
    let og = `<div></div>${labels.map(l => `<div class="colh">SLOT ${l}</div>`).join('')}`;
    [0, 1, 2].forEach(k => {
      const base = p.qwen.orders[k] || p.smol.orders[k];
      const oq = p.qwen.orders[k], os = p.smol.orders[k];
      const pq = oq && oq.pick, ps = os && os.pick;
      const miss = [show('qwen') && oq && pq == null && 'Qwen: no valid pick', show('smol') && os && ps == null && 'Smol: no valid pick'].filter(Boolean).join('<br>');
      og += `<div class="rowh"><b>Order ${k + 1}</b><span>${k === 0 ? 'original' : 'permuted'}</span>${miss ? `<div class="nopick">${miss}</div>` : ''}</div>`;
      og += base.order.map(ci => {
        const c = p.candidates[ci], best = p.best.includes(ci);
        const hq = show('qwen') && pq === ci, hs = show('smol') && ps === ci;
        return `<div class="cell"><div class="oimg ${hq ? 'pq' : ''} ${hs ? 'ps' : ''} ${best ? 'best' : ''}"><div class="img"><img src="${c.thumb}" alt="" ${retry}>${tags(hq, hs)}</div>
          <div class="cap"><span>${gen(c)}</span><b class="num">${c.utility.toFixed(2)}</b></div></div></div>`;
      }).join('');
    });
    const consLine = j => {
      const J = p[j];
      if (!J.all_valid) return `<span><b class="${j === 'qwen' ? 'q' : 's'}">${JN[j]}</b> not all orders parsed</span>`;
      const picks = J.orders.map(o => o.pick);
      const k = Math.max(...picks.map(x => picks.filter(y => y === x).length));
      return `<span><b class="${j === 'qwen' ? 'q' : 's'}">${JN[j]}</b> picks the same image in ${k} of 3 orders</span>`;
    };
    const bar = (v, col) => v == null ? '<span class="muted">–</span>' : `<div class="mbar"><div class="track"><i style="width:${100 * v}%;background:${col}"></i></div><span class="num">${v.toFixed(2)}</span></div>`;
    const rrows = p.candidates.map((c, i) => `<tr><td style="width:48px"><img class="rthumb" src="${c.thumb}" alt="" ${retry}></td>
      <td style="white-space:nowrap">${esc(c.source)}${p.best.includes(i) ? ' <span class="good">★</span>' : ''} ${p.qwen.pick0 === i ? '<span class="tag q">Q</span>' : ''}${p.smol.pick0 === i ? '<span class="tag s">S</span>' : ''}<div class="ratings num">${c.n_utility} raters: ${(c.ratings || []).map(x => x.toFixed(1)).join(', ')}</div></td>
      <td>${bar(c.utility, utilColor(c.utility))}</td><td>${bar(c.stereotype, '#8C8C8C')}</td><td>${bar(c.missing_explicit, '#8C8C8C')}</td><td>${bar(c.missing_implicit, '#8C8C8C')}</td></tr>`).join('');
    const mv = (j, key, fmt) => { const m = p[j].m0; return m ? fmt(m[key]) : '<span class="muted">no valid pick</span>'; };
    const yn = (b, good) => `<span class="${b === good ? 'good' : 'bad'}">${b ? 'yes' : 'no'}</span>`;
    const slotOf = (j, k) => { const o = p[j].orders[k]; return o && o.pick != null ? o.labels[o.order.indexOf(o.pick)] : null; };
    const pickName = j => p[j].pick0 == null ? '–' : `${esc(p.candidates[p[j].pick0].source)} <span class="muted">(slot ${slotOf(j, 0)})</span>`;
    const mrow = (label, key, fmt) => `<tr><td>${label}</td><td class="num">${mv('qwen', key, fmt)}</td><td class="num">${mv('smol', key, fmt)}</td></tr>`;
    const g = p.gates, q0 = p.qwen.m0;
    const gate = (title, ok, desc, detail) => `<div class="gate"><h3>${title}<span class="v ${ok ? 'good' : 'bad'}">${ok ? 'Accepts' : 'Rejects'}</span></h3><p>${desc}</p><p class="muted">${detail}</p></div>`;
    const qPicks = p.qwen.orders.map(o => o && o.pick);
    const unTxt = g.unanimity ? `Kept pick: regret ${f3(q0.regret)}, gain ${sg(q0.gain)}.` : `Qwen's picks by order: ${qPicks.map(i => i == null ? '–' : gen(p.candidates[i])).join(', ')}.`;
    const cmTxt = g.cross_model ? `Kept pick: regret ${f3(q0.regret)}, gain ${sg(q0.gain)}.` : `Qwen chose ${p.qwen.pick0 == null ? '–' : p.candidates[p.qwen.pick0].source}; Smol chose ${p.smol.pick0 == null ? '–' : p.candidates[p.smol.pick0].source}.`;
    const byOrder = ['qwen', 'smol'].map(j => `<tr><td><b class="${j === 'qwen' ? 'q' : 's'}">${JN[j]}</b></td>${[0, 1, 2].map(k => {
      const o = p[j].orders[k];
      if (!o || o.pick == null) return '<td class="bad">no valid pick</td>';
      const c = p.candidates[o.pick];
      return `<td style="white-space:nowrap"><b>${slotOf(j, k)}</b> <span class="muted">·</span> ${gen(c)} <span class="muted num">${c.utility.toFixed(2)}</span></td>`;
    }).join('')}</tr>`).join('');

    overlay.innerHTML = `<div class="sheet" role="dialog"><div class="sheet-in">
      <div class="dhead"><div style="min-width:0"><button class="back" id="close">← Back to prompts</button>
        <h1>${esc(p.prompt)}</h1>
        <div class="meta">${esc(p.country)} · ${esc(p.category.replace(/-/g, ' '))} · ${n} candidates · <span class="pid">${p.id.slice(0, 12)}</span></div></div>
        <div><div class="label" style="text-align:right;margin-bottom:4px">Show picks of</div>
        <div class="toggle">${['both', 'qwen', 'smol'].map(j => `<button class="${j} ${dj === j ? 'on' : ''}" data-dj="${j}">${j === 'both' ? 'Both judges' : JN[j]}</button>`).join('')}</div></div></div>
      <div class="dbody">
        <div>
          <section class="sec"><h2>The same candidates in three presentation orders</h2>
            <p class="dek">Outlines and <span class="tag q">Q</span> <span class="tag s">S</span> labels mark each judge's pick; ★ marks the human-best image; numbers are alignment ratings.</p>
            <div class="ogrid" style="grid-template-columns:78px repeat(${n}, minmax(0, 150px))">${og}</div>
            <div class="cons">${show('qwen') ? consLine('qwen') : ''}${show('smol') ? consLine('smol') : ''}</div>
          </section>
          <div class="gates">
            ${gate('Unanimity gate', g.unanimity, 'Keeps Qwen\'s pick only if it is the same in all three orders.', unTxt)}
            ${gate('Cross-model gate', g.cross_model, 'Keeps the pick only if Qwen and Smol agree in the original order.', cmTxt)}
          </div>
        </div>
        <div>
          <section class="sec"><h2>Human ratings</h2><p class="dek">Means over annotators, 0–1. Alignment is the utility (higher is better); stereotype and missing elements are lower-is-better.</p>
            <table class="t"><thead><tr><th></th><th>Generator</th><th>Alignment</th><th>Stereotype</th><th>Missing expl.</th><th>Missing impl.</th></tr></thead><tbody>${rrows}</tbody></table></section>
          <section class="sec"><h2>Selection metrics, original order</h2><p class="dek">A random pick would score ${f3(p.random_utility)} on average (regret ${f3(p.oracle_utility - p.random_utility)}); the best image scores ${f3(p.oracle_utility)}.</p>
            <table class="t"><thead><tr><th style="width:30%"></th><th class="q" style="color:${C.qwen}">Qwen</th><th style="color:${C.smol}">Smol</th></tr></thead><tbody>
              <tr><td>Pick</td><td>${pickName('qwen')}</td><td>${pickName('smol')}</td></tr>
              ${mrow('Utility', 'utility', v => f3(v))}
              ${mrow('Regret', 'regret', v => `<span class="${regretCls(v)}">${f3(v)}</span>`)}
              ${mrow('Gain over random', 'gain', v => `<span class="${signCls(v)}">${sg(v)}</span>`)}
              ${mrow('Below the mean', 'below_mean', v => yn(v, false))}
              ${mrow('Human-best', 'best', v => yn(v, true))}
            </tbody></table></section>
          <section class="sec"><h2>Picks by order</h2>
            <table class="t"><thead><tr><th style="width:18%"></th><th>Order 1</th><th>Order 2</th><th>Order 3</th></tr></thead><tbody>${byOrder}</tbody></table></section>
        </div>
      </div></div></div>`;
    overlay.querySelector('#close').onclick = () => { location.hash = exploreHash().slice(1); };
    overlay.querySelectorAll('[data-dj]').forEach(b => b.onclick = () => { state.dj = b.dataset.dj; history.replaceState(null, '', '#detail=' + p.id + (state.dj !== 'both' ? '&judge=' + state.dj : '')); renderDetail(); });
  }
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && state.detail) location.hash = exploreHash().slice(1); });

  window.addEventListener('hashchange', parseHash);
  parseHash();
})();
