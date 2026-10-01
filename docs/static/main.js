/* JudgeActs project page: order demo, slot chart, gate explorer, nav.
   Vanilla JS, no dependencies. All numbers come from static/data/*.json. */
(function () {
  'use strict';

  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var hasIO = 'IntersectionObserver' in window;
  var SVGNS = 'http://www.w3.org/2000/svg';

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function getJSON(url) {
    return fetch(url).then(function (r) { if (!r.ok) throw new Error(url + ' ' + r.status); return r.json(); });
  }
  function fmt2(x) { return x.toFixed(2); }
  function signed(x) { return (x < 0 ? '−' : '+') + Math.abs(x).toFixed(3); }
  function pct(x) { return (x * 100).toFixed(1) + '%'; }
  function svg(tag, attrs, parent) {
    var el = document.createElementNS(SVGNS, tag);
    for (var k in attrs) el.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(el);
    return el;
  }
  function textW(t, per) {
    try { var w = t.getComputedTextLength(); if (w > 0) return w; } catch (e) {}
    return t.textContent.length * per;
  }
  function onVisible(el, cb, threshold) {
    if (!hasIO) { cb(); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { io.disconnect(); cb(); } });
    }, { threshold: threshold || 0.3 });
    io.observe(el);
  }
  function setSegs(group, attr, value) {
    $$('button.seg', group).forEach(function (b) {
      var on = b.getAttribute(attr) === String(value);
      b.classList.toggle('is-on', on);
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
  }

  /* ---------------- Section reveal + active nav ---------------- */
  function initReveal() {
    var els = $$('.reveal');
    if (reduce || !hasIO) { els.forEach(function (e) { e.classList.add('in'); }); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { threshold: 0.25 });
    els.forEach(function (e) { io.observe(e); });
  }

  function initNav() {
    var links = $$('.topnav-links a');
    var map = {};
    links.forEach(function (a) { map[a.getAttribute('href').slice(1)] = a; });
    var sections = Object.keys(map).map(function (id) { return document.getElementById(id); }).filter(Boolean);
    var current = null;
    function activate(id) {
      if (id === current) return;
      current = id;
      links.forEach(function (a) { a.classList.remove('is-active'); a.removeAttribute('aria-current'); });
      var a = map[id];
      if (a) {
        a.classList.add('is-active');
        a.setAttribute('aria-current', 'true');
        var bar = a.parentNode;
        if (bar.scrollWidth > bar.clientWidth) {
          var left = a.offsetLeft - bar.clientWidth / 2 + a.offsetWidth / 2;
          bar.scrollTo({ left: left, behavior: reduce ? 'auto' : 'smooth' });
        }
      }
    }
    // Active section: the last one whose top has passed 30% of the viewport.
    var ticking = false;
    function update() {
      ticking = false;
      var line = window.innerHeight * 0.3, id = null;
      for (var i = 0; i < sections.length; i++) {
        if (sections[i].getBoundingClientRect().top <= line) id = sections[i].id;
      }
      if ((window.innerHeight + window.scrollY) >= document.documentElement.scrollHeight - 4) id = sections[sections.length - 1].id;
      activate(id);
    }
    window.addEventListener('scroll', function () {
      if (!ticking) { ticking = true; requestAnimationFrame(update); }
    }, { passive: true });
    window.addEventListener('resize', update);
    update();
  }

  /* ---------------- 1. Order demo ---------------- */
  function initDemo(data) {
    var root = $('#demo-widget');
    if (!root) return;
    var track = $('#slot-track', root);
    var big = $('#demo-big');
    var scale = $('#demo-scale');
    var dot = $('.scale-dot', scale);
    var controls = $('.demo-controls', root);
    var playBtn = $('#demo-play');
    var byKey = {};
    data.candidates.forEach(function (c) { byKey[c.key] = c; });
    var cands = {};
    $$('.cand', track).forEach(function (f) { cands[f.getAttribute('data-key')] = f; });

    // candidate ticks and random (pool mean) marker on the 0-1 scale
    var mean = data.candidates.reduce(function (s, c) { return s + c.rating; }, 0) / data.candidates.length;
    data.candidates.forEach(function (c) {
      var t = document.createElement('div');
      t.className = 'scale-tick';
      t.style.left = (c.rating * 100) + '%';
      scale.insertBefore(t, dot);
    });
    var meanEl = $('.scale-mean', scale);
    meanEl.style.left = (mean * 100) + '%';
    $('span', meanEl).textContent = 'random ' + fmt2(mean);

    var slotIndex = { A: 0, B: 1, C: 2, D: 3 };
    var state = -1, shown = 0, timer = null, playing = false, tween = null, settle = null;

    function countTo(target) {
      if (tween) cancelAnimationFrame(tween);
      if (reduce) { shown = target; big.textContent = fmt2(target); return; }
      var from = shown, t0 = null, dur = 520;
      function step(ts) {
        if (t0 === null) t0 = ts;
        var p = Math.min(1, (ts - t0) / dur);
        var e = 1 - Math.pow(1 - p, 3);
        shown = from + (target - from) * e;
        big.textContent = fmt2(shown);
        if (p < 1) tween = requestAnimationFrame(step); else { shown = target; big.textContent = fmt2(target); }
      }
      tween = requestAnimationFrame(step);
    }

    function show(i, instant) {
      if (i === state) return;
      state = i;
      var order = data.orders[i];
      setSegs(controls, 'data-order', i);
      Object.keys(order.slots).forEach(function (slot) {
        var f = cands[order.slots[slot]];
        if (f) f.style.setProperty('--i', slotIndex[slot]);
      });
      var picked = order.selected;
      var rating = byKey[picked].rating;
      var finish = function () {
        root.classList.remove('is-moving');
        Object.keys(cands).forEach(function (k) { cands[k].classList.toggle('is-picked', k === picked); });
        dot.style.left = (rating * 100) + '%';
        countTo(rating);
      };
      clearTimeout(settle);
      if (instant || reduce) { finish(); return; }
      root.classList.add('is-moving');
      settle = setTimeout(finish, 720);
    }

    function stop() {
      playing = false; clearInterval(timer); timer = null;
      playBtn.textContent = 'Play'; playBtn.setAttribute('aria-pressed', 'false');
    }
    function play() {
      playing = true;
      playBtn.textContent = 'Pause'; playBtn.setAttribute('aria-pressed', 'true');
      clearInterval(timer);
      timer = setInterval(function () { show((state + 1) % data.orders.length); }, 3000);
    }

    $$('button.seg', controls).forEach(function (b) {
      b.addEventListener('click', function () { stop(); show(+b.getAttribute('data-order')); });
    });
    playBtn.hidden = false;
    playBtn.addEventListener('click', function () {
      if (playing) { stop(); return; }
      show((state + 1) % data.orders.length);
      play();
    });

    show(0, true);
    // Auto-advance once the demo is on screen; pause while it is off screen.
    if (!reduce && hasIO) {
      var started = false;
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting && !started) {
            started = true;
            setTimeout(function () { if (!playing && state === 0) { show(1); play(); } }, 1400);
          } else if (!e.isIntersecting && playing) {
            clearInterval(timer); timer = null;
          } else if (e.isIntersecting && playing && !timer) {
            play();
          }
        });
      }, { threshold: 0.35 });
      io.observe(root);
    }
  }

  /* ---------------- 2. Slot chart ---------------- */
  function initSlots(data) {
    var host = $('#slot-chart');
    if (!host) return;
    var controls = host.closest('section').querySelector('.chart-controls');
    var caption = $('#slot-caption');
    var W, H, m, iw, ih, ymax = 0.6, narrow;
    function layout() {
      narrow = host.clientWidth < 520;
      W = narrow ? 370 : 640; H = narrow ? 290 : 300;
      m = { l: narrow ? 34 : 40, r: 4, t: 34, b: 50 };
      iw = W - m.l - m.r; ih = H - m.t - m.b;
    }
    function y(v) { return m.t + ih - (v / ymax) * ih; }
    var view = 'judges';
    var captions = {
      judges: caption.innerHTML,
      ablations: ''
    };
    var abl = data.variants.map(function (v) { return v.label + ' ' + pct(v.first); }).join(', ');
    captions.ablations = 'Qwen&rsquo;s first-slot share stays far above uniform under every exploratory prompt variant: ' +
      data.variants.map(function (v) { return v.label.toLowerCase() + ' (' + v.note + ') ' + pct(v.first) + ' against ' + pct(v.first_uniform); }).join('; ') + '.' +
      (data.omitted.length ? ' <span class="chart-note">Not shown: ' + data.omitted.map(function (o) { return o.label.toLowerCase() + ' (run incomplete, ' + o.calls + ' of ' + o.expected + ' calls)'; }).join(', ') + '.</span>' : '');

    function frame(s) {
      [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6].forEach(function (v) {
        svg('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), 'class': v === 0 ? 'ax' : 'grid' }, s);
        svg('text', { x: m.l - 8, y: y(v) + 4, 'class': 'tick', 'text-anchor': 'end' }, s).textContent = Math.round(v * 100) + '%';
      });
    }

    function drawJudges(s) {
      var slots = ['A', 'B', 'C', 'D'], gw = iw / 4, bw = narrow ? gw * 0.36 : Math.min(46, gw * 0.3);
      var models = [['qwen', 'q', 'qt'], ['smol', 's', 'st']];
      var lx = m.l;
      models.forEach(function (mm, j) {
        var md = data.models[mm[0]];
        svg('rect', { x: lx, y: 6, width: 10, height: 10, 'class': mm[1] }, s);
        var t = svg('text', { x: lx + 16, y: 15, 'class': 'leg ' + mm[2] }, s);
        t.textContent = narrow ? md.label : md.label + ' (' + md.calls + ' calls)';
        lx += 16 + textW(t, 7) + (narrow ? 14 : 22);
      });
      svg('line', { x1: lx, x2: lx + 18, y1: 11, y2: 11, 'class': 'uni' }, s);
      svg('text', { x: lx + 24, y: 15, 'class': 'leg', fill: '#5f5f5f' }, s).textContent = 'uniform';
      slots.forEach(function (sl, i) {
        var cx = m.l + gw * i + gw / 2;
        var u = data.models.qwen.uniform[sl];
        var ul = svg('line', { x1: cx - bw - 10, x2: cx + bw + 10, y1: y(u), y2: y(u), 'class': 'uni' }, s);
        models.forEach(function (mm, j) {
          var v = data.models[mm[0]].share[sl];
          var x = cx + (j === 0 ? -bw - 2 : 2);
          var r = svg('rect', { x: x, y: y(v), width: bw, height: y(0) - y(v), 'class': 'bar ' + mm[1] }, s);
          r.style.transitionDelay = (i * 90 + j * 50) + 'ms';
          svg('text', { x: x + bw / 2, y: y(v) - 6, 'class': 'val ' + mm[2] }, s).textContent = pct(v);
        });
        s.appendChild(s.removeChild(ul)); // dashes above bars
        $$('text.val', s).forEach(function (t) { s.appendChild(t); }); // labels above dashes
        svg('text', { x: cx, y: H - m.b + 22, 'class': 'xl' }, s).textContent = 'Slot ' + sl;
      });
    }

    function drawAblations(s) {
      var vs = data.variants, gw = iw / Math.max(3, vs.length), bw = Math.min(64, gw * 0.42);
      svg('rect', { x: m.l, y: 6, width: 10, height: 10, 'class': 'q' }, s);
      var t = svg('text', { x: m.l + 16, y: 15, 'class': 'leg qt' }, s);
      t.textContent = narrow ? 'Qwen3-VL-4B, slot A share' : 'Qwen3-VL-4B: share of calls choosing slot A';
      var lx = m.l + 16 + textW(t, 6.6) + 20;
      svg('line', { x1: lx, x2: lx + 18, y1: 11, y2: 11, 'class': 'uni' }, s);
      svg('text', { x: lx + 24, y: 15, 'class': 'leg', fill: '#5f5f5f' }, s).textContent = 'uniform';
      vs.forEach(function (v, i) {
        var cx = m.l + gw * i + gw / 2;
        var r = svg('rect', { x: cx - bw / 2, y: y(v.first), width: bw, height: y(0) - y(v.first), 'class': 'bar v' }, s);
        r.style.transitionDelay = (i * 110) + 'ms';
        svg('text', { x: cx, y: y(v.first) - 6, 'class': 'val qt' }, s).textContent = pct(v.first);
        svg('line', { x1: cx - bw / 2 - 10, x2: cx + bw / 2 + 10, y1: y(v.first_uniform), y2: y(v.first_uniform), 'class': 'uni' }, s);
        svg('text', { x: cx, y: H - m.b + 22, 'class': 'xl' }, s).textContent = v.label;
        svg('text', { x: cx, y: H - m.b + 39, 'class': 'xs' }, s).textContent = v.calls + ' calls';
      });
    }

    function render(animate) {
      host.classList.remove('in');
      host.innerHTML = '';
      layout();
      host.classList.toggle('is-narrow', narrow);
      var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, 'aria-hidden': 'true', focusable: 'false' }, host);
      frame(s);
      (view === 'judges' ? drawJudges : drawAblations)(s);
      caption.innerHTML = captions[view];
      host.setAttribute('aria-label', view === 'judges'
        ? 'Share of choices by slot. ' + ['qwen', 'smol'].map(function (k) {
            var d = data.models[k]; return d.label + ': ' + ['A', 'B', 'C', 'D'].map(function (sl) { return sl + ' ' + pct(d.share[sl]); }).join(', ');
          }).join('. ') + '.'
        : 'Qwen first-slot share by variant: ' + abl + '.');
      if (!animate || reduce) { host.classList.add('in'); return; }
      host.getBoundingClientRect(); // commit the scaleY(0) state before growing
      requestAnimationFrame(function () { requestAnimationFrame(function () { host.classList.add('in'); }); });
    }

    render(false);
    var seen = false;
    var lastNarrow = narrow, rt = null;
    window.addEventListener('resize', function () {
      clearTimeout(rt);
      rt = setTimeout(function () {
        if ((host.clientWidth < 520) !== lastNarrow) { lastNarrow = !lastNarrow; render(false); if (!seen && !reduce) host.classList.remove('in'); }
      }, 150);
    });
    host.classList.remove('in');
    if (reduce) host.classList.add('in');
    else onVisible(host, function () { seen = true; host.classList.add('in'); }, 0.45);

    $$('button.seg', controls).forEach(function (b) {
      b.addEventListener('click', function () {
        var v = b.getAttribute('data-view');
        if (v === view) return;
        view = v; setSegs(controls, 'data-view', v); render(true);
      });
    });
  }

  /* ---------------- 2b. Scale small multiples ---------------- */
  function initScale(data) {
    var host = $('#scale-sm');
    if (!host) return;
    var lo = -0.1, hi = 0.2, fmax = 0.6;
    function gx(v) { return ((v - lo) / (hi - lo)) * 100; }
    function fx(v) { return (v / fmax) * 100; }
    var rowsDef = [['all', 'All prompts', 'a'], ['agree', 'Orders agree', 'k'], ['disagree', 'Orders disagree', 'r']];
    host.innerHTML = '';
    data.judges.forEach(function (j, pi) {
      var p = mk('div', 'smp' + (j.key === 'mlx_8b' ? ' big8' : ''), host);
      p.style.setProperty('--p', pi);
      mk('p', 'smp-h', p, '<b>' + j.label + '</b><span>' + j.long.replace(/^Qwen3-VL-\dB, /, '') + '</span>');
      // first-slot share
      mk('p', 'smp-k', p, 'Picks the first slot');
      var f = mk('div', 'smf', p);
      var ft = mk('div', 'smf-track', f);
      var fb = mk('div', 'smf-bar', ft); fb.style.width = fx(j.first) + '%';
      var fu = mk('div', 'smf-uni', ft); fu.style.left = fx(j.first_uniform) + '%';
      mk('span', 'smf-val', f, pct(j.first));
      mk('p', 'smp-sub', p, 'uniform ' + pct(j.first_uniform) + ' &middot; reordering changes the pick on ' + pct(j.flip));
      // gains
      mk('p', 'smp-k', p, 'Gain over random (95% CI)');
      var gw = mk('div', 'smg', p);
      var ax = mk('div', 'smg-axis', gw);
      [-0.1, 0, 0.1, 0.2].forEach(function (v) {
        var gl = mk('div', v === 0 ? 'zl' : 'gl', ax); gl.style.left = gx(v) + '%';
        if (v < 0) return;
        var tl = mk('span', 'tl' + (v === 0 ? ' z' : ''), ax, v === 0 ? 'random' : '+' + v.toFixed(1));
        tl.style.left = gx(v) + '%';
      });
      var cl = mk('div', 'clip', ax); cl.style.left = gx(data.clip_gain) + '%';
      if (pi === 0) { var ct = mk('span', 'clip-l', ax, 'CLIP ' + signed(data.clip_gain)); ct.style.left = gx(data.clip_gain) + '%'; }
      rowsDef.forEach(function (rd, ri) {
        var g = j[rd[0]];
        var r = mk('div', 'smg-row ' + rd[2], gw);
        r.style.setProperty('--r', ri);
        mk('span', 'smg-lab', r, rd[1] + ' <i>' + g.n + '</i>');
        var t = mk('div', 'smg-track', r);
        var a = Math.min(0, g.gain), b = Math.max(0, g.gain);
        var bar = mk('div', 'smg-bar', t); bar.style.left = gx(a) + '%'; bar.style.width = (gx(b) - gx(a)) + '%';
        bar.style.transformOrigin = g.gain < 0 ? '100% 50%' : '0 50%';
        var ci = mk('div', 'smg-ci', t); ci.style.left = gx(g.ci[0]) + '%'; ci.style.width = (gx(g.ci[1]) - gx(g.ci[0])) + '%';
        mk('span', 'smg-val', r, signed(g.gain));
      });
    });
    if (reduce) { host.classList.add('in'); return; }
    onVisible(host, function () { requestAnimationFrame(function () { host.classList.add('in'); }); }, 0.35);
  }

  /* ---------------- 3. Gate explorer ---------------- */
  function initGates(data) {
    var grid = $('#dotgrid');
    if (!grid) return;
    var section = grid.closest('section');
    var controls = $('.chart-controls', section);
    var count = $('#gate-count'), text = $('#gate-text'), axis = $('#gaxis');
    var rows = { kept: $('.grow[data-row="kept"]'), rejected: $('.grow[data-row="rejected"]') };
    var N = data.n_prompts, dots = [];
    var lo = -0.15, hi = 0.2;
    function xp(v) { return ((v - lo) / (hi - lo)) * 100; }

    // Build grid: one row per country (30 prompts each), column-ordered.
    var frag = document.createDocumentFragment(), prev = null, col = 0;
    data.countries.forEach(function (c, i) {
      if (c !== prev) {
        var lab = document.createElement('span'); lab.className = 'cn'; var cname = c.replace(/_/g, ' '); lab.textContent = (cname === 'South Africa' && grid.clientWidth < 420) ? 'S. Africa' : cname; lab.title = cname;
        frag.appendChild(lab); prev = c; col = 0;
      }
      var d = document.createElement('span'); d.className = 'dot';
      d.style.transitionDelay = reduce ? '0ms' : (col * 14 + (i % 7)) + 'ms';
      frag.appendChild(d); dots.push(d); col++;
    });
    grid.innerHTML = ''; grid.appendChild(frag);

    // Axis
    axis.innerHTML = '';
    [-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2].forEach(function (v) {
      var gl = document.createElement('div');
      gl.className = v === 0 ? 'zero' : 'gl'; gl.style.left = xp(v) + '%'; axis.appendChild(gl);
      if (Math.abs(v * 100) % 10 === 0) {
        var tl = document.createElement('span'); tl.className = 'tl' + (v === 0 ? ' z' : '');
        tl.style.left = xp(v) + '%';
        tl.textContent = v === 0 ? 'random' : (v > 0 ? '+' : '−') + Math.abs(v).toFixed(1);
        axis.appendChild(tl);
      }
    });

    var texts = {
      none: function (g) { return 'Without a filter, Qwen acts on every prompt and beats random by ' + signed(g.kept.gain) + ' on average. The content-only CLIP scorer reaches ' + signed(data.reference.clip.gain) + '.'; },
      unanimity: function (g) { return 'The order filter keeps a prompt only if Qwen returns the same image in all three orders. Kept prompts beat random by ' + signed(g.kept.gain) + '; the prompts it rejects fall below random (' + signed(g.rejected.gain) + ').'; },
      crossmodel: function (g) { return 'The two-judge filter keeps a prompt when Qwen and SmolVLM2 pick the same image. Both favour early slots, so they agree for the wrong reason: kept prompts fall below random (' + signed(g.kept.gain) + '), rejected ones beat it (' + signed(g.rejected.gain) + ').'; },
      unanimity8b: function (g) { return 'The same order filter on Qwen3-VL-8B, which shows almost no position bias. Kept prompts beat random by ' + signed(g.kept.gain) + ', but the ' + g.rejected.n + ' it rejects still beat random by ' + signed(g.rejected.gain) + ', and the 8B without any filter reaches ' + signed(g.all.gain) + '. Here the filter mostly discards good decisions.'; }
    };

    function setRow(row, g, total, label) {
      var n = $('.gn', row), bar = $('.gbar', row), ci = $('.gci', row), val = $('.gval', row);
      if (!g) {
        row.classList.add('is-empty');
        n.textContent = '(0)';
        bar.style.left = xp(0) + '%'; bar.style.width = '0%';
        ci.style.left = xp(0) + '%'; ci.style.width = '0%';
        val.textContent = 'none';
        return;
      }
      row.classList.remove('is-empty');
      $('.glab', row).firstChild.nodeValue = label + ' ';
      n.textContent = '(' + g.n + ')';
      var a = Math.min(0, g.gain), b = Math.max(0, g.gain);
      bar.style.left = xp(a) + '%'; bar.style.width = (xp(b) - xp(a)) + '%';
      ci.style.left = xp(g.ci[0]) + '%'; ci.style.width = (xp(g.ci[1]) - xp(g.ci[0])) + '%';
      val.textContent = signed(g.gain);
    }

    var current = null;
    function show(key) {
      if (key === current) return;
      current = key;
      var g = data.gates[key];
      setSegs(controls, 'data-gate', key);
      g.keep.forEach(function (k, i) { dots[i].classList.toggle('on', !!k); });
      var kept = g.kept.n;
      count.innerHTML = 'Kept <b>' + kept + '</b> of ' + N + ' prompts (' + pct(kept / N) + ')';
      setRow(rows.kept, g.kept, N, key === 'none' ? 'All' : 'Kept');
      setRow(rows.rejected, g.rejected, N, 'Rejected');
      text.textContent = texts[key](g);
    }

    // Start from "No gate" and move to unanimity when first seen, so the change itself is visible.
    show('none');
    if (reduce || !hasIO) show('unanimity');
    else onVisible(grid, function () { setTimeout(function () { if (current === 'none' && !touched) show('unanimity'); }, 700); }, 0.5);
    var touched = false;
    $$('button.seg', controls).forEach(function (b) {
      b.addEventListener('click', function () { touched = true; show(b.getAttribute('data-gate')); });
    });
  }

  /* ---------------- Method: scrollytelling walkthrough ---------------- */
  // SmolVLM2-2.2B's choices for the example prompt, orders 1-3 (runs/smol_main.jsonl).
  var SMOL_PICKS = ['B', 'B', 'A'];
  var SLOTS = ['A', 'B', 'C', 'D'];
  var GEN_SHORT = { 'SD35': 'SD3.5', 'flux': 'Flux', 'gpt-image': 'GPT-Image' };

  function mk(tag, cls, parent, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    if (parent) parent.appendChild(e);
    return e;
  }

  // One stage = one drawing of the example prompt that can be put into any of the six states.
  function makeStage(host, ex, gates) {
    var N = gates.n_prompts, countries = gates.countries;
    var keep = gates.gates.unanimity.keep, keptN = gates.gates.unanimity.kept.n;
    // Grid order in gates.json is (country, category, prompt_id); the example prompt sits at 75.
    var EX = 75;
    if (countries[EX] !== ex.country) EX = countries.indexOf(ex.country);
    var cands = ex.candidates, K = cands.length, orders = ex.orders;
    var ratings = cands.map(function (c) { return c.rating; });
    var best = Math.max.apply(null, ratings), worst = Math.min.apply(null, ratings);
    var mean = ratings.reduce(function (a, b) { return a + b; }, 0) / K;
    var keyIdx = {}; cands.forEach(function (c, i) { keyIdx[c.key] = i; });
    function colOf(r, k) { var sl = orders[r].slots; for (var s in sl) if (sl[s] === cands[k].key) return SLOTS.indexOf(s); return k; }
    function pickCol(r) { return SLOTS.indexOf(orders[r].choice); }
    function pickIdx(r) { return keyIdx[orders[r].selected]; }

    host.innerHTML = '';
    var L = {}, step = 0, timers = [], countRaf = null;

    // --- build ---
    var rowOf = [], colIdx = [], cnames = [], prev = null, ci = -1, cc = 0;
    countries.forEach(function (c) {
      if (c !== prev) { ci++; cc = 0; prev = c; cnames.push(c.replace(/_/g, ' ')); }
      rowOf.push(ci); colIdx.push(cc++);
    });
    var NC = Math.max.apply(null, colIdx) + 1, NR = cnames.length;
    var cellLayer = mk('div', 'st-cells', host);
    var cells = [];
    for (var i = 0; i < N; i++) cells.push(mk('span', 'st-cell' + (rowOf[i] % 2 ? ' alt' : '') + (i === EX ? ' ex' : '') + (keep[i] ? ' kept' : ''), cellLayer));
    var CODES = { 'Brazil': 'BR', 'Canada': 'CA', 'Chile': 'CL', 'China': 'CN', 'Germany': 'DE', 'India': 'IN', 'Iran': 'IR', 'Japan': 'JP', 'Poland': 'PL', 'South Africa': 'ZA' };
    var clabs = cnames.map(function (n) {
      return mk('span', 'st-lab', host, '<span class="lw">' + (n === 'South Africa' ? 'S. Africa' : n) + '</span><span class="ln" title="' + n + '">' + (CODES[n] || n.slice(0, 2).toUpperCase()) + '</span>');
    });
    var gcap = mk('span', 'st-cap', host, N + ' prompts &middot; ' + NR + ' countries &times; ' + NC);
    var lead = mk('span', 'st-lead', host);
    var prompt = mk('div', 'st-prompt', host, '<span class="st-meta">' + ex.country + ' &middot; ' + ex.category + ' &middot; ' + K + ' images</span><q>' + ex.prompt + '</q>');
    var genl = cands.map(function (c) { return mk('span', 'st-gen', host, c.generator); });

    var colh = [0, 1, 2].map(function (c) { return mk('span', 'st-colh', host, 'Slot ' + SLOTS[c]); });
    var rowl = orders.map(function (o, r) { return mk('span', 'st-rowl', host, 'Order ' + (r + 1)); });

    function ratingCls(v) { return v === best ? 'good' : v === worst ? 'bad' : 'mid'; }
    var ims = []; // ims[r][k]
    orders.forEach(function (o, r) {
      ims.push(cands.map(function (c, k) {
        var w = mk('div', 'st-im' + (r ? ' clone' : ''), host);
        var img = mk('img', null, w); img.src = c.src; img.alt = ''; img.decoding = 'async';
        mk('span', 'st-tag', w, GEN_SHORT[c.key] || c.key);
        mk('span', 'st-rt ' + ratingCls(c.rating), w, fmt2(c.rating));
        return w;
      }));
    });
    var qf = orders.map(function () { var f = mk('div', 'st-fr q', host); mk('span', null, f, 'Qwen'); return f; });
    var sf = orders.map(function () { var f = mk('div', 'st-fr s', host); mk('span', null, f, 'Smol'); return f; });

    var legend = mk('div', 'st-legend', host,
      '<p><i class="sw q"></i><b class="c-q">Qwen3-VL-4B</b><br>picks slot ' + orders.map(function (o) { return o.choice; }).join(', ') + '</p>' +
      '<p><i class="sw s"></i><b class="c-s">SmolVLM2-2.2B</b><br>picks slot ' + SMOL_PICKS.join(', ') + '</p>' +
      '<p class="st-mute">Neither judge sees a rating.</p>');

    var ccap = mk('span', 'st-ccap', host, 'Qwen returned');
    var thumbs = orders.map(function (o, r) {
      var t = mk('div', 'st-th', host); var img = mk('img', null, t); img.src = cands[pickIdx(r)].src; img.alt = ''; return t;
    });
    var nes = [0, 1].map(function () { return mk('span', 'st-ne', host, '&ne;'); });
    var stamp = mk('div', 'st-stamp', host, 'Abstain');
    var note = mk('p', 'st-note', host, 'Three orders, three different images. With no agreement, the filter returns nothing for this prompt.');
    var counter = mk('div', 'st-ctr', host, '<span class="st-ctr-n"><b>' + keptN + '</b> of ' + N + ' prompts kept</span><span class="st-ctr-k"><i class="kp"></i>all orders agree <i class="me"></i>this prompt</span>');
    var ctrB = counter.querySelector('b');

    var rbox = mk('div', 'st-rbox', host, '<b>Human ratings</b><span>hidden from the judge</span><span class="st-mute">mean of 3 raters per image</span>');
    var ln = svg('svg', { 'class': 'st-ln', focusable: 'false', 'aria-hidden': 'true' }, host);
    var uid = 'stm' + Math.random().toString(36).slice(2, 7);
    var defs = svg('defs', {}, ln);
    var mkr = svg('marker', { id: uid + 'a', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 8, markerHeight: 8, markerUnits: 'userSpaceOnUse', orient: 'auto' }, defs);
    svg('path', { d: 'M0,0 L8,4 L0,8 z' }, mkr);
    var mask = svg('mask', { id: uid + 'm', maskUnits: 'userSpaceOnUse', x: 0, y: 0, width: 4000, height: 4000 }, defs);
    var lm = svg('path', { d: '', fill: 'none', stroke: '#fff', 'stroke-width': 14, 'class': 'st-lm' }, mask);
    var lg = svg('g', { mask: 'url(#' + uid + 'm)' }, ln);
    var lp = svg('path', { d: '', 'class': 'st-lp', 'marker-end': 'url(#' + uid + 'a)' }, lg);
    var llen = 0;

    var axes = orders.map(function () { return mk('span', 'st-axis', host); });
    var dots = orders.map(function () { return mk('span', 'st-dot', host); });
    var brs = orders.map(function (o, r) {
      var reg = best - cands[pickIdx(r)].rating;
      var b = mk('span', 'st-br' + (reg < 1e-9 ? ' zero' : ''), host);
      return b;
    });
    var brl = orders.map(function (o, r) {
      var reg = best - cands[pickIdx(r)].rating;
      return mk('span', 'st-brl', host, reg < 1e-9 ? 'regret 0, the best' : 'regret ' + fmt2(reg));
    });
    var vRand = mk('span', 'st-v rand', host), vBest = mk('span', 'st-v best', host);
    var lRand = mk('span', 'st-vl rand', host, 'random ' + fmt2(mean));
    var lBest = mk('span', 'st-vl best', host, 'best ' + fmt2(best));
    var end0 = mk('span', 'st-end', host, '0'), end1 = mk('span', 'st-end', host, '1');
    var take = mk('p', 'st-take', host, 'Same judge, same images. Only the order changed, and the returned image went from the worst to the best.');

    // --- geometry ---
    function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
    function layout() {
      var W = host.clientWidth, H = host.clientHeight;
      if (!W || !H) return false;
      var wide = W >= 600, p = wide ? 20 : 10;
      L = { W: W, H: H, wide: wide, p: p };
      // step 1: prompt grid + example pool
      var gl = wide ? 80 : 24, capH = wide ? 24 : 20;
      prompt.style.width = Math.min(W - 2 * p, 560) + 'px';
      var ph = prompt.offsetHeight || 48;
      var g1 = wide ? 16 : 8, genH = 22;
      var s1 = Math.min(wide ? 176 : 118, (W - 2 * p - 2 * g1) / 3);
      var rest = capH + 22 + ph + 12 + genH;
      // shrink the grid before the pool images get small
      var pitch = Math.min((W - 2 * p - gl) / NC, wide ? 22 : 12, Math.max(6, (H - 2 * p - rest - Math.min(s1, H * 0.3)) / NR));
      var gridW = gl + NC * pitch;
      L.gx0 = (W - gridW) / 2; L.gx = L.gx0 + gl; L.pitch = pitch; L.c1 = Math.max(3, pitch - (wide ? 3 : 2));
      var fixed = rest + NR * pitch;
      s1 = Math.max(40, Math.min(s1, H - 2 * p - fixed));
      var total = fixed + s1;
      var top = Math.max(p, (H - total) / 2);
      L.capY = top; L.gy = top + capH; L.gridB = L.gy + NR * pitch;
      L.promptX = (W - Math.min(W - 2 * p, 560)) / 2; L.promptY = L.gridB + 22;
      L.s1 = s1; L.g1 = g1; L.poolY = L.promptY + ph + 12; L.poolX = (W - (3 * s1 + 2 * g1)) / 2;
      // steps 2-6: three order rows
      var Lw = wide ? 66 : 54, g = wide ? 12 : 6, hh = 22;
      var Rw = wide ? clamp(W * 0.27, 180, 230) : 0;
      var bb = wide ? 64 : Math.max(100, H * 0.25);
      var Wl = W - 2 * p - (wide ? Rw + 30 : 0);
      var s = Math.floor(Math.min((Wl - Lw - 2 * g) / 3, (H - 2 * p - hh - bb - 2 * g) / 3, 150));
      var rowsW = Lw + 3 * s + 2 * g, blockH = hh + 3 * s + 2 * g;
      L.s = s; L.g = g; L.hh = hh; L.Lw = Lw; L.Rw = Rw;
      var left = wide ? p + Math.max(0, (Wl - rowsW) / 2) : (W - rowsW) / 2;
      L.rx = left; L.x0 = left + Lw;
      L.y0 = (wide ? Math.max(p, (H - bb - blockH) / 2) : p) + hh;
      L.rowsR = L.x0 + 3 * s + 2 * g; L.rowsB = L.y0 + 3 * s + 2 * g; L.rowsM = (L.y0 + L.rowsB) / 2;
      L.panX = wide ? L.rowsR + 30 : p; L.panW = wide ? W - p - L.panX : W - 2 * p;
      L.bandY = L.rowsB + 14;
      // mini grid (step 4)
      if (wide) {
        L.th = Math.min(58, (L.panW - 2 * 24) / 3);
        L.thY = L.y0 + 2;
        L.mp = Math.min(7, L.panW / NC);
      } else {
        L.th = clamp((H - p - L.bandY) * 0.42, 26, 40);
        L.thY = L.bandY + 16;
        L.mp = 4;
      }
      // scale (step 6)
      L.sx0 = L.x0 + s + (wide ? 34 : 18);
      L.sx1 = W - p - (wide ? 40 : 14);
      return true;
    }
    function imRect(r, c) { return { x: L.x0 + c * (L.s + L.g), y: L.y0 + r * (L.s + L.g), w: L.s, h: L.s }; }
    function sx(v) { return L.sx0 + v * (L.sx1 - L.sx0); }

    // --- helpers ---
    function put(e, x, y, w, h, extra) {
      e.style.transform = 'translate(' + Math.round(x * 10) / 10 + 'px,' + Math.round(y * 10) / 10 + 'px)' + (extra || '');
      if (w != null) e.style.width = Math.max(0, w) + 'px';
      if (h != null) e.style.height = Math.max(0, h) + 'px';
    }
    function vis(e, on, delay, op) {
      e.style.transitionDelay = (delay || 0) + 'ms';
      e.style.opacity = on ? (op == null ? 1 : op) : 0;
      e.classList.toggle('on', !!on);
    }
    function later(fn, ms) { timers.push(setTimeout(fn, ms)); }
    function countUp(target, ms) {
      if (countRaf) cancelAnimationFrame(countRaf);
      var t0 = null;
      function f(ts) {
        if (t0 === null) t0 = ts;
        var q = Math.min(1, (ts - t0) / ms), e = 1 - Math.pow(1 - q, 3);
        ctrB.textContent = Math.round(target * e);
        if (q < 1) countRaf = requestAnimationFrame(f);
      }
      ctrB.textContent = '0';
      countRaf = requestAnimationFrame(f);
    }
    function routeLine() {
      var d;
      if (L.wide) {
        var bx = L.panX, by = L.rowsM;
        d = 'M' + (bx - 2) + ' ' + by + ' H' + (L.rowsR + 10);
      } else {
        var cx = L.W / 2, ty = L.bandY + 12;
        d = 'M' + cx + ' ' + ty + ' V' + (L.rowsB + 6);
      }
      lp.setAttribute('d', d); lm.setAttribute('d', d);
      try { llen = Math.ceil(lp.getTotalLength()) + 12; } catch (e) { llen = 400; }
      lm.style.strokeDasharray = llen + ' ' + llen;
    }

    // --- states ---
    function apply(st, instant) {
      var prevStep = step;
      step = st;
      timers.forEach(clearTimeout); timers = [];
      host.className = host.className.replace(/\bs\d\b/g, '').trim() + ' s' + st;
      var fwd = st > prevStep;
      var s = L.s, wide = L.wide;

      // prompt grid cells
      var kMini = (L.mp * 0.78) / L.c1;
      cells.forEach(function (e, i) {
        var r = rowOf[i], c = colIdx[i];
        e.style.width = e.style.height = L.c1 + 'px';
        var d = 0;
        if (st <= 1) {
          put(e, L.gx + c * L.pitch, L.gy + r * L.pitch, null, null, st === 0 ? ' scale(.2)' : '');
          d = fwd && prevStep === 0 ? c * 11 + r * 16 : (i % 23) * 6;
          vis(e, st === 1, d);
        } else if (st === 4) {
          vis(e, true, 120 + c * 9 + r * 4); // placed below, once the counter is measured
        } else {
          vis(e, false, (i % 17) * 5);
        }
      });
      host.classList.toggle('ex-on', st === 1);
      clabs.forEach(function (e, r) {
        put(e, L.gx0, L.gy + r * L.pitch + L.pitch / 2 - 7, (L.gx - L.gx0) - (L.wide ? 8 : 5));
        vis(e, st === 1, fwd ? 200 + r * 30 : 0);
      });
      put(gcap, L.gx0, L.capY); vis(gcap, st === 1, fwd ? 100 : 0);
      var exX = L.gx + colIdx[EX] * L.pitch, exY = L.gy + rowOf[EX] * L.pitch;
      put(lead, exX + L.c1 / 2 - 0.5, exY + L.c1 + 2, 1, Math.max(0, L.promptY - exY - L.c1 - 6));
      vis(lead, st === 1, st === 1 && fwd ? 700 : 0);
      put(prompt, L.promptX, L.promptY);
      vis(prompt, st === 1, st === 1 && fwd ? 760 : 0);
      genl.forEach(function (e, k) {
        put(e, L.poolX + k * (L.s1 + L.g1), L.poolY + L.s1 + 6, L.s1);
        vis(e, st === 1, st === 1 ? 1150 + k * 80 : 0);
      });

      // images
      ims.forEach(function (row, r) {
        row.forEach(function (e, k) {
          var col = colOf(r, k), R, op = 1, on = true, d = 0;
          if (st === 0) { R = { x: exX, y: exY, w: L.c1, h: L.c1 }; on = false; }
          else if (st === 1) {
            R = { x: L.poolX + k * (L.s1 + L.g1), y: L.poolY, w: L.s1, h: L.s1 };
            on = r === 0;
            d = r === 0 && fwd ? 820 + k * 110 : 0;
          } else {
            R = imRect(r, col);
            if (st === 2) d = fwd ? r * 170 + col * 40 : 0;
            if (st === 4) op = 0.42;
            if (st === 6 && col !== pickCol(r)) on = false;
          }
          put(e, R.x, R.y, R.w, R.h);
          vis(e, on, d, op);
          e.classList.toggle('big', st === 1);
        });
      });
      // the first appearance of the pool grows out of the example square
      if (st === 1 && prevStep === 0 && !instant) {
        ims[0].forEach(function (e) { e.style.transition = 'none'; put(e, exX, exY, L.c1, L.c1); e.style.opacity = 0; });
        host.getBoundingClientRect();
        ims[0].forEach(function (e, k) {
          e.style.transition = '';
          put(e, L.poolX + k * (L.s1 + L.g1), L.poolY, L.s1, L.s1);
          vis(e, true, 820 + k * 110);
        });
      }
      host.classList.toggle('rated', st >= 5);
      ims.forEach(function (row, r) {
        row.forEach(function (e, k) {
          var rt = e.querySelector('.st-rt');
          rt.style.transitionDelay = st === 5 && fwd && !instant ? (650 + (r * 3 + colOf(r, k)) * 70) + 'ms' : '0ms';
        });
      });

      colh.forEach(function (e, c) {
        var R = imRect(0, c);
        put(e, R.x, L.y0 - L.hh, s);
        vis(e, st >= 2 && st <= 5 || (st === 6 && c === 0), st === 2 && fwd ? 300 + c * 60 : 0);
      });
      rowl.forEach(function (e, r) {
        var R = imRect(r, 0);
        put(e, L.rx, R.y + s / 2 - 9, L.Lw - 8);
        vis(e, st >= 2, st === 2 && fwd ? 200 + r * 170 : 0);
      });

      // judge frames
      orders.forEach(function (o, r) {
        var R = imRect(r, pickCol(r)), q = qf[r];
        var dQ = 3, dS = 7;
        var onQ = st >= 3;
        put(q, R.x - dQ, R.y - dQ, s + 2 * dQ, s + 2 * dQ, onQ ? '' : ' scale(1.22)');
        vis(q, onQ, st === 3 && fwd ? 150 + r * 160 : 0, st === 4 ? 0.35 : 1);
        q.classList.toggle('dim', st === 4);
        var Rs = imRect(r, SLOTS.indexOf(SMOL_PICKS[r])), sfr = sf[r];
        var onS = st === 3;
        put(sfr, Rs.x - dS, Rs.y - dS, s + 2 * dS, s + 2 * dS, onS ? '' : ' scale(1.22)');
        vis(sfr, onS, st === 3 && fwd ? 750 + r * 160 : 0);
      });
      if (wide) put(legend, L.panX, L.y0, L.panW);
      else put(legend, L.p, L.bandY, L.W - 2 * L.p);
      vis(legend, st === 3, st === 3 && fwd ? 1100 : 0);

      // agreement filter
      var thGap = wide ? 24 : 18, thX0 = wide ? L.panX : L.p;
      put(ccap, thX0, L.thY - 18);
      vis(ccap, st === 4, st === 4 ? 200 : 0);
      thumbs.forEach(function (e, r) {
        var on = st === 4, R;
        if (on) R = { x: thX0 + r * (L.th + thGap), y: L.thY, w: L.th, h: L.th };
        else { var P = imRect(r, pickCol(r)); R = { x: P.x, y: P.y, w: s, h: s }; }
        put(e, R.x, R.y, R.w, R.h);
        vis(e, on, on ? 120 + r * 120 : 0);
      });
      nes.forEach(function (e, j) {
        put(e, thX0 + (j + 1) * L.th + j * thGap, L.thY + L.th / 2 - 10, thGap);
        vis(e, st === 4, st === 4 ? 600 + j * 80 : 0);
      });
      var stX = wide ? thX0 + (3 * L.th + 2 * thGap) / 2 : thX0 + 3 * L.th + 2 * thGap + 14;
      var stY = L.thY + L.th / 2;
      put(stamp, stX, stY, null, null, wide ? ' translate(-50%,-50%) rotate(-7deg)' + (st === 4 ? '' : ' scale(1.7)') : ' translate(0,-50%) rotate(-7deg)' + (st === 4 ? '' : ' scale(1.7)'));
      vis(stamp, st === 4, st === 4 ? 950 : 0);
      if (wide) {
        put(note, L.panX, L.thY + L.th + 16, L.panW);
        vis(note, st === 4, st === 4 ? 1150 : 0);
        L.ctrY = L.thY + L.th + 16 + note.offsetHeight + 18;
      } else {
        vis(note, false);
        L.ctrY = L.thY + L.th + 10;
      }
      if (wide) { put(counter, L.panX, L.ctrY, L.panW); L.miniY = L.ctrY + counter.offsetHeight + 8; }
      else { put(counter, L.p, L.ctrY, L.W - 2 * L.p - NC * L.mp - 12); L.miniY = L.ctrY + 2; L.panX = L.W - L.p - NC * L.mp; }
      vis(counter, st === 4, st === 4 ? 300 : 0);
      if (st === 4) {
        cells.forEach(function (e, i) { put(e, L.panX + colIdx[i] * L.mp, L.miniY + rowOf[i] * L.mp, null, null, ' scale(' + kMini + ')'); });
        if (!instant && prevStep !== 4) countUp(keptN, 1100); else ctrB.textContent = keptN;
      }
      if (!wide) L.panX = L.p;
      host.classList.toggle('mini', st === 4);
      host.classList.toggle('narrow', !wide);

      // ratings reveal
      if (wide) put(rbox, L.panX, L.rowsM - 38, L.panW);
      else put(rbox, (L.W - Math.min(260, L.W - 2 * L.p)) / 2, L.bandY + 12, Math.min(260, L.W - 2 * L.p));
      vis(rbox, st === 5, st === 5 && fwd ? 80 : 0);
      routeLine();
      lm.style.transitionDelay = st === 5 && !instant ? '350ms' : '0ms';
      lm.style.strokeDashoffset = st === 5 ? 0 : llen;
      ln.classList.toggle('on', st === 5);

      // scoring scale
      var on6 = st === 6;
      orders.forEach(function (o, r) {
        var R = imRect(r, 0), y = R.y + s * 0.4;
        var v = cands[pickIdx(r)].rating;
        put(axes[r], L.sx0, y, L.sx1 - L.sx0, null, on6 ? '' : ' scaleX(0)');
        vis(axes[r], on6, on6 && !instant ? 150 + r * 120 : 0);
        put(dots[r], on6 ? sx(v) : L.sx0, y);
        vis(dots[r], on6, on6 && !instant ? 350 + r * 120 : 0);
        var x0 = sx(v), x1 = sx(best);
        put(brs[r], x0, y + 15, x1 - x0, null, on6 ? '' : ' scaleX(0)');
        vis(brs[r], on6, on6 && !instant ? 1100 + r * 140 : 0);
        put(brl[r], (x0 + x1) / 2, y + 21, null, null, ' translateX(-50%)');
        vis(brl[r], on6, on6 && !instant ? 1300 + r * 140 : 0);
      });
      var vy0 = L.y0 - 4, vy1 = L.rowsB;
      put(vRand, sx(mean), vy0, null, vy1 - vy0, on6 ? '' : ' scaleY(0)');
      put(vBest, sx(best), vy0, null, vy1 - vy0, on6 ? '' : ' scaleY(0)');
      vis(vRand, on6, on6 && !instant ? 700 : 0); vis(vBest, on6, on6 && !instant ? 800 : 0);
      put(lRand, sx(mean), L.y0 - L.hh, null, null, ' translateX(-50%)');
      if (wide) put(lBest, sx(best), L.y0 - L.hh, null, null, ' translateX(-50%)');
      else put(lBest, sx(best), L.rowsB + 3, null, null, ' translateX(-50%)');
      vis(lRand, on6, on6 && !instant ? 800 : 0); vis(lBest, on6, on6 && !instant ? 900 : 0);
      var ly = imRect(2, 0).y + s * 0.4 - 22;
      put(end0, L.sx0, ly, null, null, ' translateX(-50%)'); put(end1, L.sx1, ly, null, null, ' translateX(-50%)');
      vis(end0, on6, on6 ? 500 : 0); vis(end1, on6, on6 ? 500 : 0);
      if (wide) put(take, L.rx, Math.min(L.H - L.p - 44, L.rowsB + 26), L.W - L.p - L.rx);
      else put(take, L.p, L.rowsB + 24, L.W - 2 * L.p);
      vis(take, on6, on6 && !instant ? 1700 : 0);
    }

    function set(st, instant) {
      if (!layout()) { step = st; return; }
      if (instant) host.classList.add('instant');
      apply(st, instant);
      if (instant) { host.getBoundingClientRect(); host.classList.remove('instant'); }
    }
    function relayout() {
      if (!layout()) return;
      host.classList.add('instant');
      var st = step; step = st; apply(st, true);
      host.getBoundingClientRect();
      host.classList.remove('instant');
    }
    set(0, true);
    if ('ResizeObserver' in window) {
      var lastW = 0, lastH = 0;
      new ResizeObserver(function () {
        if (host.clientWidth === lastW && host.clientHeight === lastH) return;
        lastW = host.clientWidth; lastH = host.clientHeight; relayout();
      }).observe(host);
    } else window.addEventListener('resize', relayout);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(relayout);
    return { go: function (st, instant) { if (st !== step) set(st, instant); }, step: function () { return step; } };
  }

  function initScrolly(ex, gates) {
    var root = $('#scrolly');
    if (!root) return;
    var host = $('#st-stage'), steps = $$('.st-step', root), count = $('#st-count'), allBtn = $('#st-all');
    $$('[data-fill="kept"]', root).forEach(function (e) { e.textContent = gates.gates.unanimity.kept.n; });
    var stage = makeStage(host, ex, gates);
    var cur = 0, io = null, statics = null, isStatic = false;
    var mq = window.matchMedia('(max-width: 860px)');

    function setActive(n) {
      if (n === cur) return;
      cur = n;
      stage.go(n);
      steps.forEach(function (s) { s.classList.toggle('is-active', +s.getAttribute('data-step') === n); });
      count.textContent = 'Step ' + n + ' of ' + steps.length;
    }
    function line() { return mq.matches ? 0.82 : 0.5; }
    function observe() {
      if (io) io.disconnect();
      if (!hasIO || isStatic) return;
      var top = Math.round(line() * 100);
      io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) { if (e.isIntersecting) setActive(+e.target.closest('.st-step').getAttribute('data-step')); });
      }, { rootMargin: '-' + top + '% 0px -' + (99 - top) + '% 0px', threshold: 0 });
      // Phones: the stage covers the top of the screen, so watch the cards in the window below it.
      steps.forEach(function (s) { io.observe(mq.matches ? $('.st-card', s) : s); });
    }
    function scrollToStep(n) {
      n = Math.max(1, Math.min(steps.length, n));
      var r = $('.st-card', steps[n - 1]).getBoundingClientRect();
      var target = window.scrollY + r.top + r.height / 2 - window.innerHeight * line();
      window.scrollTo({ top: target, behavior: reduce ? 'auto' : 'smooth' });
      setActive(n);
    }
    $$('.st-btn', root).forEach(function (b) {
      b.addEventListener('click', function () { scrollToStep((cur || 0) + +b.getAttribute('data-dir')); });
    });
    root.addEventListener('keydown', function (e) {
      if (isStatic || e.target.tagName === 'A') return;
      if (e.key === 'ArrowRight') { e.preventDefault(); scrollToStep(cur + 1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); scrollToStep(cur - 1); }
    });

    function buildStatics() {
      statics = steps.map(function (s) {
        var n = +s.getAttribute('data-step');
        var h = mk('div', 'st-stage st-static', s);
        h.setAttribute('aria-hidden', 'true');
        return { n: n, host: h, stage: null };
      });
    }
    function setStatic(on) {
      isStatic = on;
      root.classList.toggle('is-static', on);
      allBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
      allBtn.textContent = on ? 'Back to scrolling story' : 'Show all steps';
      if (on) {
        if (!statics) buildStatics();
        statics.forEach(function (o) {
          if (!o.stage) o.stage = makeStage(o.host, ex, gates);
          o.stage.go(o.n, true);
        });
        if (io) io.disconnect();
      } else {
        observe();
      }
    }
    allBtn.addEventListener('click', function () {
      setStatic(!isStatic);
      root.scrollIntoView({ block: 'start', behavior: 'auto' });
    });
    // In static mode the toggle lives in the first card.
    var topToggle = mk('button', 'st-all st-all-top', null, 'Back to scrolling story');
    topToggle.type = 'button';
    root.insertBefore(topToggle, root.firstChild);
    topToggle.addEventListener('click', function () { setStatic(false); root.scrollIntoView({ block: 'start' }); });

    if (reduce || !hasIO) { setStatic(true); return; }
    observe();
    // Start the first scene as soon as the stage comes into view.
    var io0 = new IntersectionObserver(function (entries) {
      if (entries[0].isIntersecting) { io0.disconnect(); if (!cur && !isStatic) setActive(1); }
    }, { threshold: 0.3 });
    io0.observe(host);
    if (mq.addEventListener) mq.addEventListener('change', observe);
  }

  /* ---------------- BibTeX copy ---------------- */
  function initCopy() {
    var btn = $('#bib-copy'), code = $('#bib-code');
    if (!btn || !code || !navigator.clipboard) return;
    btn.hidden = false;
    btn.addEventListener('click', function () {
      navigator.clipboard.writeText(code.textContent).then(function () {
        btn.textContent = 'Copied';
        setTimeout(function () { btn.textContent = 'Copy'; }, 1600);
      });
    });
  }

  function boot() {
    initReveal();
    initNav();
    initCopy();
    var exP = getJSON('static/data/example.json'), gatesP = getJSON('static/data/gates.json');
    exP.then(initDemo).catch(function (e) { console.error(e); });
    getJSON('static/data/slots.json').then(initSlots).catch(function (e) { console.error(e); });
    gatesP.then(initGates).catch(function (e) { console.error(e); });
    getJSON('static/data/scale.json').then(initScale).catch(function (e) { console.error(e); });
    Promise.all([exP, gatesP]).then(function (r) { initScrolly(r[0], r[1]); }).catch(function (e) { console.error(e); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
