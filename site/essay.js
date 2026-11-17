// The essay's concept figures, adapted from the maintainer's course notes. Each draws into a
// container on index.qmd if it is there. Teacher's text and probabilities are violet, the student's
// teal, everywhere. The token explorer and the grading strip show measurements (assets/forecast.json,
// assets/grading.json, written by fc-probe); the prefix and two-peak figures are illustrations, and
// say so in their captions.
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  function $(id) { return document.getElementById(id); }
  function el(name, attrs, text) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function h(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function pct(p) { return p < 0.0005 ? "< 0.1%" : (100 * p).toFixed(1) + "%"; }

  // Vertical bars, one group per label, one bar per series.
  function bars(svg, labels, series, opts) {
    opts = opts || {};
    var W = 600, H = opts.height || 150, padB = 24, padT = 16;
    var max = opts.max || 1;
    svg.setAttribute("viewBox", "0 0 " + W + " " + (H + padB + padT));
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var n = labels.length, group = W / n, bw = Math.max(2, (group * 0.72) / series.length);
    [0.25, 0.5, 0.75, 1].forEach(function (g) {
      if (g > max) return;
      var y = H + padT - (H * g) / max;
      svg.appendChild(el("line", { x1: 0, x2: W, y1: y, y2: y, stroke: "var(--rule)", "stroke-dasharray": "2,3" }));
      svg.appendChild(el("text", { x: W, y: y - 3, "text-anchor": "end", "font-size": 10, fill: "var(--muted)" }, Math.round(100 * g) + "%"));
    });
    svg.appendChild(el("line", { x1: 0, x2: W, y1: H + padT, y2: H + padT, stroke: "var(--ink-2)" }));
    labels.forEach(function (lab, i) {
      series.forEach(function (s, j) {
        var v = s.values[i] || 0, bh = (H * Math.min(v, max)) / max;
        var x = i * group + group * 0.14 + j * bw;
        var r = el("rect", { x: x, y: H + padT - bh, width: bw - 2, height: Math.max(0, bh), fill: s.color, rx: 1.5 });
        r.appendChild(el("title", {}, s.name + " · " + lab + ": " + pct(v)));
        svg.appendChild(r);
      });
      if (lab !== "") {
        svg.appendChild(el("text", { x: i * group + group / 2, y: H + padT + 17, "text-anchor": "middle",
          "font-size": n > 12 ? 10 : 12, fill: "var(--ink-2)", "font-family": "var(--mono)" }, lab));
      }
    });
  }

  // Segmented buttons: [label, value] pairs; calls back with the value, marks the pressed one.
  function seg(box, items, onPick) {
    box.innerHTML = "";
    items.forEach(function (it) {
      var b = h("button", null, it[0]);
      b.type = "button"; b.dataset.v = it[1];
      b.addEventListener("click", function () { onPick(it[1]); });
      box.appendChild(b);
    });
    return function (v) { box.querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.v === String(v))); }); };
  }

  // 1. The token explorer: real 2B and 0.8B probabilities at three positions of one answer.
  function soft(entry, T) {
    var top = entry.top.map(function (x) { return Math.exp(x[1] / T); });
    var rest = entry.rest.reduce(function (s, b) { return s + b[1] * Math.exp(b[0] / T); }, 0);
    var Z = top.reduce(function (a, b) { return a + b; }, 0) + rest;
    return { labels: entry.top.map(function (x) { return x[0]; }), p: top.map(function (v) { return v / Z; }), rest: rest / Z };
  }
  function kl(p, q) {
    var s = 0;
    for (var i = 0; i < p.p.length; i++) if (p.p[i] > 0) s += p.p[i] * Math.log(p.p[i] / q.p[i]);
    if (p.rest > 0) s += p.rest * Math.log(p.rest / q.rest);
    return Math.max(0, s);
  }
  function topk(entry, T, k) {
    var all = entry.top.map(function (x) { return [x[1], 1]; }).concat(entry.rest.map(function (b) { return [b[0], b[1]]; }));
    all.sort(function (a, b) { return b[0] - a[0]; });
    var Z = all.reduce(function (s, x) { return s + x[1] * Math.exp(x[0] / T); }, 0), got = 0, left = k;
    for (var i = 0; i < all.length && left > 0; i++) { var n = Math.min(left, all[i][1]); got += n * Math.exp(all[i][0] / T); left -= n; }
    return got / Z;
  }
  function tokenExplorer(P) {
    var root = $("w-token");
    if (!root) return;
    var q = function (t) { return "“" + t + "”"; };
    var shorten = function (l) { return l.length > 6 ? l.slice(0, 6) + "…" : l; };
    var top1 = function (k) { return soft(P.positions[k]["2B"], 1).p[0]; };
    // Every position where the teacher hesitates, and the first where it is sure, in answer order.
    var all = Object.keys(P.positions).sort(function (a, b) { return a - b; });
    var unsure = all.filter(function (k) { return top1(k) < 0.95; });
    var sure = all.filter(function (k) { return top1(k) >= 0.95 && P.positions[k].tok.trim().length > 2; })[0];
    var keys = unsure.concat(sure ? [sure] : []).sort(function (a, b) { return a - b; });
    var ans = root.querySelector(".tokline"), svg = root.querySelector("svg");
    var press = seg(root.querySelector(".seg"), keys.map(function (k) { return [P.positions[k].tok.trim() || P.positions[k].tok, k]; }), draw);
    function draw(k) {
      var pos = P.positions[k], t = soft(pos["2B"], 1), s = soft(pos["0.8B"], 1), n = 5;
      var y = t.labels.indexOf(pos.tok);
      bars(svg, t.labels.slice(0, n).map(shorten).concat(["others"]), [
        { values: t.p.slice(0, n).concat([t.rest]), color: "var(--teacher)", name: "teacher, 2B" },
        { values: s.p.slice(0, n).concat([s.rest]), color: "var(--student)", name: "student, 0.8B untrained" }
      ], { height: 140, max: 1 });
      ans.innerHTML = "";
      P.answer_tokens.forEach(function (tok, i) { ans.appendChild(h("span", String(i) === k ? "cur" : null, tok)); });
      root.querySelector("[data-o=hard]").textContent = (-Math.log(s.p[y])).toFixed(3);
      root.querySelector("[data-o=soft]").textContent = kl(t, s).toFixed(3);
      var kept = topk(pos["2B"], 1, 20);
      root.querySelector("[data-o=top]").textContent = kept > 0.999 ? "> 99.9%" : pct(kept);
      root.querySelector(".note").textContent = t.p[0] >= 0.95
        ? "The teacher is sure: " + pct(t.p[0]) + " on " + q(t.labels[0].trim()) + ". Hard and soft targets are almost the same here, and so are the two losses."
        : "The teacher writes " + q(t.labels[0].trim()) + " (" + pct(t.p[0]) + ") but gives " + q(t.labels[1].trim()) + " " + pct(t.p[1]) +
          ". B's target says " + q(t.labels[0].trim()) + " and nothing else; C's keeps the other alive. The untrained student puts " + pct(s.p[0]) + " on " + q(t.labels[0].trim()) + ".";
      press(k);
    }
    var start = unsure.slice().sort(function (a, b) { return top1(a) - top1(b); })[0] || keys[0];
    draw(start);
  }

  // 2. Which prefixes does training ever see? An illustration of compounding error.
  function prefixes() {
    var root = $("w-prefix");
    if (!root) return;
    var teacher = ['[{', '"name"', ':', ' "weather"', ',', ' "arguments"', ':', ' {', '"city"', ':', ' "Porto"', '}}', ']'];
    var student = ['[{', '"name"', ':', ' "weather"', ',', ' "arguments"', ']', ',', ' {', '"city"', ':', ' "Porto"', '}]'];
    var SLIP = 6, box = root.querySelector(".toks"), note = root.querySelector(".note");
    function line(label, toks, cls) {
      var row = h("div", "tokrow");
      row.appendChild(h("span", "who", label));
      toks.forEach(function (t, i) { row.appendChild(h("span", "tok " + cls(i), t)); });
      box.appendChild(row);
    }
    var press = seg(root.querySelector(".seg"), [["B and C: the teacher's text", "off"], ["D: the student's text", "on"]], show);
    function show(mode) {
      box.innerHTML = "";
      if (mode === "off") {
        line("trained on", teacher, function () { return "t"; });
        line("writes at test", student, function (i) { return (i < SLIP ? "seen" : "unseen") + (i === SLIP ? " err" : ""); });
        note.textContent = "Grey: prefixes training covered. Amber: after the slip, every prefix is one the teacher never wrote, so B and C never trained on it.";
      } else {
        line("trained on", student, function (i) { return "s" + (i === SLIP ? " err" : ""); });
        note.textContent = "The slip is in D's training data. The teacher finds that early bracket very unlikely, so the student is pushed away from exactly the mistake it makes.";
      }
      press(mode);
    }
    show("off");
  }

  // 3. The teacher grades every token of the student's own answer (assets/grading.json, measured).
  function grading(G) {
    var root = $("w-grade");
    if (!root) return;
    root.querySelector(".lead").innerHTML = "";
    root.querySelector(".lead").append("The untrained 0.8B's answer to ", h("em", null, "“" + G.query + "”"), ". Click a token:");
    root.querySelector(".gold").textContent = "gold: " + JSON.stringify(G.gold);
    var toks = G.tokens.map(function (t) { return [t.tok, Math.exp(t.student), Math.exp(t.teacher), t.student - t.teacher]; });
    var row = root.querySelector(".tokrow");
    var press = seg(row, toks.map(function (t, i) { return [t[0], i]; }), pick);
    row.querySelectorAll("button").forEach(function (b, i) {
      var k = toks[i][3], heat = Math.min(1, Math.abs(k) / 2.3);
      b.style.background = "color-mix(in srgb, " + (k > 0 ? "var(--bad)" : "var(--good)") + " " + Math.round(4 + 56 * heat) + "%, var(--surface))";
    });
    function pick(i) {
      i = +i;
      var t = toks[i], k = t[3];
      root.querySelector("[data-o=ps]").textContent = t[1].toFixed(3);
      root.querySelector("[data-o=pt]").textContent = t[2].toFixed(3);
      root.querySelector("[data-o=k]").textContent = (k >= 0 ? "+" : "") + k.toFixed(2);
      root.querySelector(".note").textContent = k > 1
        ? "The student was far surer of this token than the teacher: the loss pushes it down hard."
        : k > 0.3 ? "The student was surer than the teacher: the loss pushes this token down a little."
        : k < -0.3 ? "The teacher liked this token more than the student did: the loss pushes it up."
        : "They agree: almost nothing to learn here.";
      press(i);
    }
    var worst = toks.reduce(function (m, t, i) { return t[3] > toks[m][3] ? i : m; }, 0);
    pick(worst);
  }

  // 4. Fit one peak to a two-peaked teacher: forward against reverse KL.
  function twoPeaks() {
    var root = $("w-peaks");
    if (!root) return;
    var N = 21, labels = [];
    for (var i = 0; i < N; i++) labels.push(i % 5 === 0 ? String(i) : "");
    function norm(a) { var s = a.reduce(function (x, y) { return x + y; }, 0); return a.map(function (x) { return x / s; }); }
    function gauss(mu, sd) { var a = []; for (var i = 0; i < N; i++) a.push(Math.exp(-0.5 * Math.pow((i - mu) / sd, 2)) + 1e-6); return norm(a); }
    var teacher = norm(gauss(5, 1.3).map(function (v, i) { return v + gauss(15, 1.3)[i]; }));
    function div(p, q) { var s = 0; for (var i = 0; i < N; i++) s += p[i] * Math.log(p[i] / q[i]); return s; }
    var mu = root.querySelector("[name=mu]"), sd = root.querySelector("[name=sd]");
    function draw() {
      var student = gauss(+mu.value, +sd.value);
      bars(root.querySelector("svg"), labels, [
        { values: teacher, color: "var(--teacher)", name: "teacher" },
        { values: student, color: "var(--student)", name: "student" }
      ], { height: 130, max: 0.35 });
      root.querySelector("[data-o=fwd]").textContent = div(teacher, student).toFixed(2);
      root.querySelector("[data-o=rev]").textContent = div(student, teacher).toFixed(2);
    }
    [mu, sd].forEach(function (x) { x.addEventListener("input", draw); });
    draw();
  }

  function start() {
    prefixes(); twoPeaks();
    var base = (document.querySelector('meta[name="quarto:offset"]') || {}).content || "./";
    var load = function (name, draw) { fetch(base + "assets/" + name).then(function (r) { return r.json(); }).then(draw); };
    if ($("w-token")) load("forecast.json", tokenExplorer);
    if ($("w-grade")) load("grading.json", grading);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
