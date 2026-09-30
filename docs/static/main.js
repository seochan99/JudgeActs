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
    captions.ablations = 'Qwen&rsquo;s first-slot share stays far above uniform under every post-hoc variant: ' +
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
      none: function (g) { return 'Without a gate, Qwen acts on every prompt and beats random by ' + signed(g.kept.gain) + ' on average. The content-only CLIP scorer reaches ' + signed(data.reference.clip.gain) + '.'; },
      unanimity: function (g) { return 'Order unanimity keeps a prompt only if Qwen returns the same image in all three orders. Kept prompts beat random by ' + signed(g.kept.gain) + '; the prompts it rejects fall below random (' + signed(g.rejected.gain) + ').'; },
      crossmodel: function (g) { return 'Cross-model agreement keeps a prompt when Qwen and SmolVLM2 pick the same image. Both favour early slots, so they agree for the wrong reason: kept prompts fall below random (' + signed(g.kept.gain) + '), rejected ones beat it (' + signed(g.rejected.gain) + ').'; }
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
    getJSON('static/data/example.json').then(initDemo).catch(function (e) { console.error(e); });
    getJSON('static/data/slots.json').then(initSlots).catch(function (e) { console.error(e); });
    getJSON('static/data/gates.json').then(initGates).catch(function (e) { console.error(e); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
