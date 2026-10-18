/* Jobintel front-end behaviour.  No framework: forms show a loading state, filters submit when a sort changes, Save buttons post to the API, Plotly figures are drawn from JSON blocks.
   Every page works without this file (forms are plain GET/POST); the script only makes it smoother. */
(function () {
  "use strict";
  var csrf = document.body.getAttribute("data-csrf") || "";

  // ---- a polite live message at the bottom of the screen (replaces alert()) ----
  var toastBox = null, toastTimer = null;
  function toast(message, kind) {
    if (!toastBox) {
      toastBox = document.createElement("div");
      toastBox.className = "toast-live"; toastBox.setAttribute("role", "status"); toastBox.setAttribute("aria-live", "polite");
      document.body.appendChild(toastBox);
    }
    toastBox.innerHTML = "";
    var n = document.createElement("div");
    n.className = "note " + (kind === "bad" ? "note-bad" : kind === "good" ? "note-good" : "");
    n.textContent = message;
    toastBox.appendChild(n);
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toastBox.innerHTML = ""; }, 5000);
  }

  // ---- loading state: any form with data-loading="message" shows the veil + disables its submit buttons ----
  var veil = document.getElementById("loading-veil");
  function clearLoading() {
    if (veil) { veil.classList.remove("on"); veil.setAttribute("aria-hidden", "true"); }
    document.querySelectorAll("button[type=submit]").forEach(function (b) {
      if (b.hasAttribute("data-was-disabled")) return;
      b.disabled = false; var s = b.querySelector(".spinner-border"); if (s) s.remove();
    });
  }
  function startLoading(form) {
    var msg = form.getAttribute("data-loading") || "";
    var title = form.getAttribute("data-loading-title") || "Working";
    if (veil) {
      document.getElementById("loading-title").textContent = title;
      document.getElementById("loading-text").textContent = msg;
      veil.classList.add("on"); veil.setAttribute("aria-hidden", "false");
    }
    form.querySelectorAll("button[type=submit]").forEach(function (b) {
      b.disabled = true;
      b.insertAdjacentHTML("afterbegin", '<span class="spinner-border" aria-hidden="true"></span>');
    });
  }
  document.querySelectorAll("form[data-loading]").forEach(function (form) {
    form.addEventListener("submit", function () { if (form.checkValidity()) startLoading(form); });
  });
  window.addEventListener("pageshow", function (e) { if (e.persisted) clearLoading(); });          // back button: clear a stale veil

  // ---- a select with data-autosubmit submits its form (given by the form attribute or the closest form) when changed ----
  document.querySelectorAll("select[data-autosubmit]").forEach(function (sel) {
    sel.addEventListener("change", function () {
      var form = sel.form || document.getElementById(sel.getAttribute("form"));
      if (form) { if (form.requestSubmit) form.requestSubmit(); else form.submit(); }
    });
  });
  document.querySelectorAll(".no-js-only").forEach(function (el) { el.hidden = true; });

  // ---- Plotly figures from <script type="application/json" id=...> (dark theme) ----
  var PALETTE = ["#d9a441", "#9bbd97", "#dc9082", "#8fb4c9", "#b9a88a", "#c9a0b4", "#a8a298", "#7f9f7a"];
  function drawCharts(root) {
    if (!window.Plotly) return;
    (root || document).querySelectorAll("[data-chart]").forEach(function (el) {
      if (el.getAttribute("data-drawn") === "1") return;
      var src = document.getElementById(el.getAttribute("data-chart"));
      if (!src) return;
      var spec; try { spec = JSON.parse(src.textContent); } catch (e) { return; }
      var grid = { gridcolor: "#2e2b27", zerolinecolor: "#4a463f", linecolor: "#4a463f", tickfont: { color: "#a8a298" } };
      var layout = Object.assign({
        font: { family: "Atkinson Hyperlegible Next Variable, Segoe UI, system-ui, sans-serif", size: 13, color: "#f3efe8" },
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", colorway: PALETTE,
        margin: { l: 56, r: 16, t: spec.layout && spec.layout.title ? 40 : 12, b: 48 }, legend: { orientation: "h", y: -0.22 }
      }, spec.layout || {});
      layout.xaxis = Object.assign({}, grid, layout.xaxis || {});
      layout.yaxis = Object.assign({}, grid, layout.yaxis || {});
      var h = spec.height; if (h) el.style.minHeight = h + "px";
      window.Plotly.newPlot(el, spec.data || [], layout, { responsive: true, displayModeBar: false });
      el.setAttribute("data-drawn", "1");
    });
  }
  drawCharts();
  document.addEventListener("shown.bs.tab", function (e) {      // charts in hidden tabs need a resize once visible
    var pane = document.querySelector(e.target.getAttribute("data-bs-target"));
    if (pane) { drawCharts(pane); pane.querySelectorAll("[data-chart]").forEach(function (el) { if (window.Plotly && el.getAttribute("data-drawn") === "1") window.Plotly.Plots.resize(el); }); }
  });

  // ---- Save / Mark-as-applied buttons (logged-in users): POST JSON, flip the button ----
  document.querySelectorAll("button[data-interact]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var on = btn.classList.contains("btn-state-on");
      var action = btn.getAttribute("data-interact");
      var body = JSON.stringify({ job_id: parseInt(btn.getAttribute("data-job"), 10), action: on ? "un" + action : action });
      btn.disabled = true; btn.setAttribute("data-was-disabled", "1");
      fetch(btn.getAttribute("data-url"), { method: "POST", headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf }, body: body, credentials: "same-origin" })
        .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
        .then(function (res) {
          if (!res.ok) { toast(res.j.error || "That did not work. Try again.", "bad"); return; }
          btn.classList.toggle("btn-state-on", !on);
          var lbl = btn.querySelector(".lbl"); (lbl || btn).textContent = btn.getAttribute(on ? "data-off" : "data-on");
          toast(on ? btn.getAttribute("data-msg-off") || "Removed." : btn.getAttribute("data-msg-on") || "Done.", "good");
        }).catch(function () { toast("Could not reach the server. Check your connection and try again.", "bad"); })
        .finally(function () { btn.disabled = false; btn.removeAttribute("data-was-disabled"); });
    });
  });

  // ---- weight sliders show their value (used by the recommendation page) ----
  document.querySelectorAll("input[type=range][data-out]").forEach(function (r) {
    var out = document.getElementById(r.getAttribute("data-out"));
    function upd() { if (out) out.textContent = Number(r.value).toFixed(2); }
    r.addEventListener("input", upd); upd();
  });
  var reset = document.getElementById("reset-weights");
  if (reset) reset.addEventListener("click", function () {
    document.querySelectorAll("input[type=range][data-default]").forEach(function (r) { r.value = r.getAttribute("data-default"); r.dispatchEvent(new Event("input")); });
  });

  // ---- lazy buttons: GET json -> fill a target element ----
  document.querySelectorAll("button[data-fetch]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var target = document.getElementById(btn.getAttribute("data-target"));
      btn.disabled = true; btn.textContent = btn.getAttribute("data-busy") || "Loading";
      fetch(btn.getAttribute("data-fetch"), { credentials: "same-origin" }).then(function (r) { return r.json(); }).then(function (j) {
        if (target) target.innerHTML = j.html || "";
        btn.remove();
      }).catch(function () { btn.disabled = false; btn.textContent = "Try again"; });
    });
  });
})();
