// The brain view: every neuron as a point, lit as it spikes (spec section 11).
// Reads /meta, /positions.bin and /classes.bin once, then frames from WS /activity.

import * as THREE from "three";
import { OrbitControls } from "./OrbitControls.js";

const TAU_MS = 100;         // brain-time decay of a neuron's glow (spec section 11)
const IDLE_TAU_S = 0.4;     // wall-time fade once frames stop
const IDLE_AFTER_S = 2;     // no frame for this long: the game is between episodes or stopped
const SPARK_MS = 5000;      // sparkline window, brain time
const GROUP_COLORS = {
  pr_L: "#b6a6ff", pr_R: "#b6a6ff", lc10a_L: "#3fd0c0", lc10a_R: "#3fd0c0",
  turn_neg: "#4cc4ff", turn_pos: "#ff6b9e", fwd: "#8fdc6a", mdn: "#c78bff",
  pam: "#ffd75e", ppl1: "#e0844a", kc: "#6f7cff", mbon: "#3fe0a0",
};
const DEFAULT_OFF = new Set(["pam", "ppl1", "kc", "mbon"]);
const SPARKS = [["turn_neg", "#4cc4ff"], ["turn_pos", "#ff6b9e"], ["fwd", "#8fdc6a"],
                ["mdn", "#c78bff"], ["pop_hz", "#ffb454"]];

const $ = (id) => document.getElementById(id);

async function fetchOk(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r;
}

async function load() {
  const meta = await (await fetchOk("meta")).json();
  const pos = new Float32Array(await (await fetchOk("positions.bin")).arrayBuffer());
  const cls = new Uint8Array(await (await fetchOk("classes.bin")).arrayBuffer());
  if (pos.length !== meta.n * 3 || cls.length !== meta.n) throw new Error("data size mismatch");
  return { meta, pos, cls };
}

function classColor(i) {
  const c = new THREE.Color();
  c.setHSL(((i * 0.618034) % 1 + 0.55) % 1, 0.28, 0.55);
  return c;
}

const VERT = `
  attribute vec3 color; attribute float size; attribute float hl; attribute float act;
  uniform float uScale; uniform float uSize; uniform float uBase;
  varying vec3 vColor; varying float vAlpha;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float a = 1.0 - exp(-0.8 * act);
    vec3 hot = mix(vec3(1.0, 0.71, 0.33), vec3(1.0), hl);
    vColor = mix(color, hot, a);
    vAlpha = max(mix(uBase, 0.85, hl), a);
    gl_PointSize = max(uSize * size * (1.0 + 1.2 * a) * uScale / -mv.z, 1.0);
    gl_Position = projectionMatrix * mv;
  }`;
const FRAG = `
  varying vec3 vColor; varying float vAlpha;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = dot(c, c);
    if (d > 0.25) discard;
    float f = smoothstep(0.25, 0.02, d);
    gl_FragColor = vec4(vColor, vAlpha * f);
  }`;

function main({ meta, pos, cls }) {
  const n = meta.n, cond = meta.condition;
  const stage = $("stage");

  // Scene, centered on the cloud. Positions are micrometres (MaleCNS cache convention): +x is
  // the fly's left, y points ventral, small z is anterior. The default camera sits behind the
  // fly (+z, posterior) looking toward the origin with up = (0, -1, 0), so dorsal is up and
  // screen-right is the fly's right.
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let i = 0; i < n; i++) box.expandByPoint(v.set(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]));
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1);
  const centered = new Float32Array(pos.length);
  for (let i = 0; i < n; i++) {
    centered[3 * i] = pos[3 * i] - center.x;
    centered[3 * i + 1] = pos[3 * i + 1] - center.y;
    centered[3 * i + 2] = pos[3 * i + 2] - center.z;
  }

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x0c0f14);
  stage.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, radius / 100, radius * 20);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  function resetView() {
    camera.up.set(0, -1, 0);
    camera.position.set(0, 0, radius * 2.6);
    controls.target.set(0, 0, 0);
    controls.update();
  }
  resetView();
  window.addEventListener("keydown", (e) => { if (e.key === "r" || e.key === "R") resetView(); });

  const base = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  const sizes = new Float32Array(n).fill(1);
  const hl = new Float32Array(n);
  const act = new Float32Array(n);
  const palette = meta.classes.map((_, i) => classColor(i));
  for (let i = 0; i < n; i++) {
    const c = palette[cls[i]] || palette[0];
    base[3 * i] = c.r; base[3 * i + 1] = c.g; base[3 * i + 2] = c.b;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(centered, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.setAttribute("size", new THREE.BufferAttribute(sizes, 1));
  geo.setAttribute("hl", new THREE.BufferAttribute(hl, 1));
  const actAttr = new THREE.BufferAttribute(act, 1);
  actAttr.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute("act", actAttr);
  const uniforms = { uScale: { value: 1 }, uSize: { value: radius * 0.004 },
                     uBase: { value: 0.05 } };
  const mat = new THREE.ShaderMaterial({ uniforms, vertexShader: VERT, fragmentShader: FRAG,
                                         transparent: true, depthWrite: false,
                                         blending: THREE.AdditiveBlending });
  scene.add(new THREE.Points(geo, mat));

  // Highlight groups: toggles, colors, sizes and floating labels.
  const groups = meta.groups.filter((g) => g.idx.length > 0);
  const labels = $("labels");
  for (const g of groups) {
    g.color = new THREE.Color(GROUP_COLORS[g.key] || "#ffffff");
    g.on = !DEFAULT_OFF.has(g.key);
    const c = new THREE.Vector3();
    for (const i of g.idx) c.add(v.set(centered[3 * i], centered[3 * i + 1], centered[3 * i + 2]));
    g.center = c.divideScalar(g.idx.length);
    g.tag = document.createElement("div");
    g.tag.className = g.dn ? "tag dn" : "tag";
    g.tag.style.color = GROUP_COLORS[g.key] || "#fff";
    g.tag.textContent = g.label;
    g.dot = document.createElement("i");
    g.tag.appendChild(g.dot);
    labels.appendChild(g.tag);
  }
  function paint() {
    colors.set(base); sizes.fill(1); hl.fill(0);
    // DN groups last, so a neuron in two groups shows as the DN.
    for (const g of [...groups.filter((x) => !x.dn), ...groups.filter((x) => x.dn)]) {
      g.tag.style.display = g.on ? "" : "none";
      if (!g.on) continue;
      for (const i of g.idx) {
        colors[3 * i] = g.color.r; colors[3 * i + 1] = g.color.g; colors[3 * i + 2] = g.color.b;
        sizes[i] = g.dn ? 7 : 2.2;
        hl[i] = 1;
      }
    }
    for (const k of ["color", "size", "hl"]) geo.attributes[k].needsUpdate = true;
  }
  const list = $("groups");
  for (const g of groups) {
    const row = document.createElement("label");
    row.innerHTML = `<input type="checkbox"><span class="sw"></span><span></span>`;
    row.querySelector("input").checked = g.on;
    row.querySelector(".sw").style.background = GROUP_COLORS[g.key] || "#fff";
    row.lastChild.textContent = `${g.label} (${g.idx.length})`;
    row.querySelector("input").addEventListener("change", (e) => { g.on = e.target.checked; paint(); });
    list.appendChild(row);
  }
  paint();
  $("dim").addEventListener("input", (e) => { uniforms.uBase.value = +e.target.value; });
  const classBox = $("classes");
  meta.classes.forEach((name, i) => {
    const sw = document.createElement("span");
    sw.className = "sw";
    sw.style.background = `#${palette[i].getHexString()}`;
    const t = document.createElement("span");
    t.textContent = name;
    classBox.append(sw, t);
  });

  // The condition: wiring is the first thing to read.
  const wiring = $("wiring");
  wiring.textContent = cond.wiring;
  if (cond.brain_id.startsWith("fly-scrambled")) wiring.classList.add("scrambled");
  $("cond").textContent = `plasticity ${cond.plasticity ? "on" : "off"} · rung ${cond.rung} · ` +
    `w_scale ${cond.w_scale} · ${cond.fingerprint.slice(0, 12)} · ${n.toLocaleString()} neurons`;
  if (cond.decoder.fwd_silent) {
    const fwdLabel = meta.groups.find((g) => g.key === "fwd").label;
    $("fwdnote").textContent = `${fwdLabel} stayed silent in calibration, so forward speed is fixed.`;
  }

  // Panel instruments.
  const eye = $("eye").getContext("2d");
  const eyeImg = eye.createImageData(72, 30);
  const sparkBox = $("sparks");
  const groupLabel = Object.fromEntries(groups.map((g) => [g.key, g.label]));
  const sparks = SPARKS.map(([key, color]) => {
    const row = document.createElement("div");
    row.className = "spark";
    row.innerHTML = `<span></span><canvas></canvas><span class="v">-</span>`;
    row.firstChild.textContent = key === "pop_hz" ? "population" : (groupLabel[key] || key);
    row.firstChild.style.color = color;
    sparkBox.appendChild(row);
    return { key, color, canvas: row.querySelector("canvas"), out: row.lastChild, t: [], y: [] };
  });
  function drawSpark(s, now) {
    const cv = s.canvas, dpr = Math.min(window.devicePixelRatio, 2);
    const w = cv.clientWidth * dpr, h = cv.clientHeight * dpr;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const g = cv.getContext("2d");
    g.clearRect(0, 0, w, h);
    g.strokeStyle = "#222a35"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, h - 0.5); g.lineTo(w, h - 0.5); g.stroke();
    if (s.t.length < 2) return;
    const top = Math.max(5, ...s.y) * 1.1;
    const x = (t) => ((t - (now - SPARK_MS)) / SPARK_MS) * w;
    const y = (val) => h - 1 - (val / top) * (h - 3);
    g.beginPath();
    g.moveTo(x(s.t[0]), h);
    for (let i = 0; i < s.t.length; i++) g.lineTo(x(s.t[i]), y(s.y[i]));
    g.lineTo(x(s.t[s.t.length - 1]), h);
    g.closePath();
    g.fillStyle = s.color + "33"; g.fill();
    g.beginPath();
    for (let i = 0; i < s.t.length; i++) g[i ? "lineTo" : "moveTo"](x(s.t[i]), y(s.y[i]));
    g.strokeStyle = s.color; g.lineWidth = 1.5 * dpr; g.stroke();
    const lx = x(s.t[s.t.length - 1]), ly = y(s.y[s.y.length - 1]);
    g.fillStyle = s.color; g.beginPath(); g.arc(lx, ly, 2.5 * dpr, 0, 7); g.fill();
  }

  // Frames.
  let last = null, lastWall = 0, dirty = false, glowing = false;
  function onFrame(buf) {
    const dv = new DataView(buf);
    const k = dv.getUint32(0, true);
    const info = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, k)));
    const counts = new Uint8Array(buf, 4 + k, n);
    const pix = new Uint8Array(buf, 4 + k + n, 72 * 30);
    const fresh = !last || info.episode !== last.episode || info.t < last.t;
    const f = fresh ? 0 : Math.exp(-(info.t - last.t) / TAU_MS);
    for (let i = 0; i < n; i++) act[i] = act[i] * f + counts[i];
    dirty = glowing = true;
    lastWall = performance.now();

    for (let i = 0; i < 72 * 30; i++) {
      const p = pix[i], o = 4 * i;
      eyeImg.data[o] = eyeImg.data[o + 1] = eyeImg.data[o + 2] = p;
      eyeImg.data[o + 3] = 255;
    }
    eye.putImageData(eyeImg, 0, 0);

    $("episode").textContent = info.episode < 0 ? "-" : info.episode;
    $("step").textContent = info.step;
    $("score").textContent = Math.round(info.score);
    if (last && info.episode === last.episode && info.score > last.score) {
      const b = $("scorebox");
      b.classList.remove("flash"); void b.offsetWidth; b.classList.add("flash");
    }
    const lim = cond.max_dtheta || 64;
    const frac = Math.max(-1, Math.min(1, info.dtheta / lim));
    const fill = $("turnfill");
    fill.style.left = `${50 + Math.min(frac, 0) * 50}%`;
    fill.style.width = `${Math.abs(frac) * 50}%`;
    fill.style.background = frac < 0 ? "var(--left)" : "var(--right)";
    $("dtheta").textContent = `${info.dtheta >= 0 ? "+" : ""}${info.dtheta.toFixed(1)}°`;
    $("turnhz").textContent = `signal ${info.turn.toFixed(1)} Hz`;
    $("speedfill").style.width = `${Math.max(0, Math.min(1, info.speed)) * 100}%`;
    $("speedfill").style.background = "var(--fwd)";
    $("speed").textContent = info.speed.toFixed(2);
    $("clock").textContent = `brain ${(info.t / 1000).toFixed(2)} s`;

    for (const s of sparks) {
      if (fresh) { s.t.length = 0; s.y.length = 0; }
      const val = s.key === "pop_hz" ? info.pop_hz : (info.rates[s.key] ?? 0);
      s.t.push(info.t); s.y.push(val);
      while (s.t.length && s.t[0] < info.t - SPARK_MS) { s.t.shift(); s.y.shift(); }
      s.out.textContent = `${val.toFixed(1)} Hz`;
    }
    last = info;
    setConn("live", "live");
  }

  const conn = $("conn");
  function setConn(cls, text) { conn.className = `pill ${cls}`; conn.textContent = text; }
  let socket = null;
  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/activity`);
    socket = ws;
    ws.binaryType = "arraybuffer";
    ws.onopen = () => setConn("idle", "waiting for the game");
    ws.onmessage = (e) => onFrame(e.data);
    ws.onclose = async () => {
      setConn("down", "brain offline, retrying");
      await new Promise((r) => setTimeout(r, 1000));
      try {  // a brain restarted with other wiring, or any other condition change, needs a
             // fresh page
        const m = await (await fetchOk("meta")).json();
        if (JSON.stringify(m.condition) !== JSON.stringify(cond) || m.n !== n) {
          location.reload();
          return;
        }
      } catch { /* still down: connect() retries */ }
      connect();
    };
  }
  connect();

  // Render loop.
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
    uniforms.uScale.value = (h * renderer.getPixelRatio()) / (2 * Math.tan((camera.fov * Math.PI) / 360));
  }
  new ResizeObserver(resize).observe(stage);
  resize();
  let prev = performance.now();
  const p = new THREE.Vector3();
  function tick(now) {
    const dt = (now - prev) / 1000;
    prev = now;
    if (glowing && now - lastWall > IDLE_AFTER_S * 1000) {
      if (socket && socket.readyState === WebSocket.OPEN) setConn("idle", "waiting for the game");
      const f = Math.exp(-dt / IDLE_TAU_S);
      let peak = 0;
      for (let i = 0; i < n; i++) { act[i] *= f; if (act[i] > peak) peak = act[i]; }
      dirty = true;
      if (peak < 0.01) { act.fill(0); glowing = false; }
    }
    if (dirty) { actAttr.needsUpdate = true; dirty = false; }
    controls.update();
    renderer.render(scene, camera);
    const w = stage.clientWidth, h = stage.clientHeight;
    for (const g of groups) {
      if (!g.on) continue;
      p.copy(g.center).project(camera);
      const show = p.z < 1 && Math.abs(p.x) < 1.2 && Math.abs(p.y) < 1.2;
      g.tag.style.visibility = show ? "visible" : "hidden";
      g.tag.style.left = `${(p.x + 1) * w / 2}px`;
      g.tag.style.top = `${(1 - p.y) * h / 2}px`;
      let a = 0;
      for (const i of g.idx) a = Math.max(a, act[i]);
      g.dot.style.background = a > 0.05 ? "currentColor" : "transparent";
      g.dot.style.opacity = a > 0.05 ? Math.min(1, 0.4 + a / 2) : 1;
    }
    const t = last ? last.t : 0;
    for (const s of sparks) drawSpark(s, t);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

async function boot() {
  for (;;) {
    let data;
    try { data = await load(); }
    catch (e) {
      $("conn").className = "pill down";
      $("conn").textContent = `cannot load the brain (${e.message}), retrying`;
      await new Promise((r) => setTimeout(r, 2000));
      continue;
    }
    return main(data);
  }
}
boot();
