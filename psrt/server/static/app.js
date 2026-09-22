/* Piston System Research Tool -- browser front end.
 *
 * Everything here reads and writes the same design state the CLI does. There
 * is no separate web model: the rail renders /api/state, an edit posts back,
 * and a refused write surfaces the reason the constraint layer gave.
 *
 * Frame convention, matching psrt.geometry.build: model +Z is the cylinder
 * axis and the pin axis is model Y. The whole assembly lives in one group
 * rotated -90 degrees about X, so model +Z points up on screen and the
 * default orbit controls behave.
 */
'use strict';

const S = {
  state: null, sweep: null, margins: null, metrics: null,
  meshes: {}, group: null, arrows: {}, clip: null, fea: null,
  envelope: null,
  idx: 0, framed: false, playing: false, mode: 'assembly', showForces: true, showSleeve: true,
};

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v))
  ? '-' : Number(v).toFixed(d);

async function api(path, options) {
  const r = await fetch(path, options);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}
const post = (path, body) => api(path, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: body === undefined ? undefined : JSON.stringify(body),
});

function toast(message, refused) {
  const el = $('toast');
  el.innerHTML = message;
  el.className = 'toast show' + (refused ? ' refused' : '');
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.className = 'toast'; }, refused ? 7000 : 3200);
}
const note = (t) => { $('status').textContent = t || ''; };

/* ------------------------------------------------------------------ rail */

const UNIT_DISPLAY = {
  m: ['mm', 1e3], 'm^2': ['mm2', 1e6], 'm^3': ['cc', 1e6], 'm^4': ['mm4', 1e12],
  Pa: ['bar', 1e-5], kg: ['g', 1e3], N: ['kN', 1e-3], W: ['kW', 1e-3],
  'rad/s': ['rpm', 60 / (2 * Math.PI)], rad: ['deg', 180 / Math.PI],
  K: ['C', 1], 'J/kg': ['MJ/kg', 1e-6],
};
const toDisplay = (v, u) => (u === 'K') ? v - 273.15
  : (UNIT_DISPLAY[u] ? v * UNIT_DISPLAY[u][1] : v);
const toSI = (v, u) => (u === 'K') ? v + 273.15
  : (UNIT_DISPLAY[u] ? v / UNIT_DISPLAY[u][1] : v);
const unitLabel = (u) => UNIT_DISPLAY[u] ? UNIT_DISPLAY[u][0] : (u || '');

function renderRail(state) {
  $('engine-name').textContent = state.name;
  $('engine-note').textContent = state.notes || state.provenance || '';
  $('undo').disabled = !state.can_undo;

  const open = new Set([...document.querySelectorAll('details.group[open]')]
    .map((d) => d.dataset.section));
  if (!open.size) { ['engine', 'piston', 'rod'].forEach((s) => open.add(s)); }

  const host = $('params');
  host.innerHTML = '';
  for (const group of state.sections) {
    const d = document.createElement('details');
    d.className = 'group';
    d.dataset.section = group.section;
    if (open.has(group.section)) d.open = true;
    d.innerHTML = `<summary>${group.section}</summary>`;

    for (const p of group.parameters) {
      const row = document.createElement('div');
      row.className = 'param';
      const tag = p.class === 'locked' ? '<span class="tag locked">locked</span>'
        : p.class === 'derived' ? '<span class="tag derived">derived</span>'
        : p.class === 'bounded' ? '<span class="tag bounded">bounded</span>' : '';
      const numeric = typeof p.value === 'number';
      const shown = numeric ? toDisplay(p.value, p.unit) : p.value;
      const label = unitLabel(p.unit);

      row.innerHTML =
        `<div class="label" title="${(p.description || '').replace(/"/g, '&quot;')}">`
        + `${p.name.replace(/_/g, ' ')}${tag}</div>`
        + `<input value="${numeric ? Number(shown).toPrecision(6).replace(/\.?0+$/, '') : (shown ?? '')}"`
        + `${p.editable ? '' : ' disabled'} data-path="${p.path}" data-unit="${p.unit}">`;

      if (p.editable && label) {
        const r = document.createElement('div');
        r.className = 'range';
        const lo = p.min !== null ? fmt(toDisplay(p.min, p.unit), 3) : '';
        const hi = p.max !== null ? fmt(toDisplay(p.max, p.unit), 3) : '';
        r.textContent = (lo || hi) ? `${lo || '-'} .. ${hi || '-'} ${label}` : label;
        row.appendChild(r);
      }
      const why = p.class === 'locked' ? p.why : (p.why_max || p.why_min);
      if (why) {
        const w = document.createElement('div');
        w.className = 'why';
        w.textContent = why;
        row.appendChild(w);
      }
      d.appendChild(row);
    }
    host.appendChild(d);
  }

  host.querySelectorAll('input[data-path]').forEach((input) => {
    input.addEventListener('change', async () => {
      const path = input.dataset.path;
      const unit = input.dataset.unit;
      const raw = Number(input.value);
      if (Number.isNaN(raw)) { refresh(); return; }
      await write({ [path]: toSI(raw, unit) }, `${path} edited in the rail`);
    });
  });
}

async function write(changes, rationale) {
  note('evaluating...');
  const result = await post('/api/state', { changes, rationale });
  if (!result.ok) {
    const v = result.violation;
    toast(`<strong>Refused:</strong> ${v.path} &mdash; ${v.reason}`, true);
    note('');
    await refresh();
    return false;
  }
  const worse = Object.entries(result.deltas || {})
    .filter(([k, d]) => /peak_pin|side_thrust|reciprocating|peak_pressure/.test(k)
      && d.delta > 0)
    .map(([k, d]) => `${k.split('.').pop()} ${d.percent > 0 ? '+' : ''}${fmt(d.percent, 1)}%`);
  if (worse.length) toast('Also got worse: ' + worse.slice(0, 3).join(', '));
  await refresh();
  return true;
}

/* --------------------------------------------------------------- margins */

function band(sf) {
  if (sf === null) return 'good';
  if (sf < 1) return 'critical';
  if (sf < 1.25) return 'serious';
  if (sf < 1.5) return 'warning';
  return 'good';
}

function renderMargins(report) {
  const worst = report.margins.slice(0, 10);
  const full = 3.0;

  $('binding-note').innerHTML = report.binding_operating
    ? `Worst constraint that moves with the operating point: <strong style="color:var(--text-secondary)">${report.binding_operating}</strong>`
    : '';

  $('margin-chart').innerHTML = worst.map((m) => {
    const sf = m.safety_factor;
    const pct = sf === null ? 100 : Math.min(sf / full, 1) * 100;
    const b = band(sf);
    return `<div class="mbar">
      <div class="row"><span class="who">${m.component} &middot; ${m.mode}</span>
      <span class="sf" style="color:var(--${b})">${sf === null ? 'inf' : fmt(sf)}</span></div>
      <div class="track"><div class="fill" style="width:${pct}%;background:var(--${b})"></div>
      <div class="limit" style="left:${(1 / full) * 100}%"></div></div>
    </div>`;
  }).join('') + `<p style="font-size:10px;color:var(--text-muted);margin:2px 0 0">
    marker at safety factor 1.0; scale ends at ${full.toFixed(1)}</p>`;

  $('margins').innerHTML = worst.slice(0, 6).map((m) => `
    <div class="margin-item">
      <span class="pill ${band(m.safety_factor)}">${band(m.safety_factor) === 'critical' ? 'FAILS' : 'ok'}</span>
      <span class="mode">${m.component} &mdash; ${m.mode}</span>
      <div class="meta">${m.condition}</div>
      <div class="eq">${m.equation}</div>
    </div>`).join('');
}

/* ---------------------------------------------------------------- charts */

const CSS = (name) => getComputedStyle(document.documentElement)
  .getPropertyValue(name).trim();

function chartHeight() {
  // Leave the assembly room to breathe on a short window.
  const v = window.innerHeight;
  return v < 760 ? 96 : v < 900 ? 118 : 140;
}

function drawChart(canvas, xs, series, cursor) {
  const dpr = window.devicePixelRatio || 1;
  const h = chartHeight();
  canvas.style.width = '100%';
  canvas.style.height = h + 'px';
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  const padL = 38, padR = 8, padT = 8, padB = 18;
  const plotW = w - padL - padR, plotH = h - padT - padB;

  let lo = 0, hi = 0;
  for (const s of series) for (const v of s.data) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (hi === lo) hi = lo + 1;
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;

  const X = (i) => padL + (i / (xs.length - 1)) * plotW;
  const Y = (v) => padT + (1 - (v - lo) / (hi - lo)) * plotH;

  // Recessive grid, then a stronger zero line where one exists.
  g.strokeStyle = CSS('--grid'); g.lineWidth = 1;
  for (let k = 0; k <= 3; k++) {
    const v = lo + (hi - lo) * (k / 3), y = Math.round(Y(v)) + 0.5;
    g.beginPath(); g.moveTo(padL, y); g.lineTo(w - padR, y); g.stroke();
    g.fillStyle = CSS('--text-muted');
    g.font = '10px ui-monospace, monospace'; g.textAlign = 'right';
    g.fillText(v.toFixed(Math.abs(hi) < 10 ? 1 : 0), padL - 5, y + 3);
  }
  if (lo < 0 && hi > 0) {
    g.strokeStyle = CSS('--baseline'); g.lineWidth = 1;
    const y = Math.round(Y(0)) + 0.5;
    g.beginPath(); g.moveTo(padL, y); g.lineTo(w - padR, y); g.stroke();
  }

  g.textAlign = 'center'; g.fillStyle = CSS('--text-muted');
  for (const deg of [-360, -180, 0, 180, 360]) {
    const i = xs.findIndex((v) => v >= deg);
    if (i < 0) continue;
    g.fillText(deg + '°', X(i), h - 5);
  }

  for (const s of series) {
    g.strokeStyle = s.color; g.lineWidth = 2;
    g.lineJoin = 'round'; g.lineCap = 'round';
    g.beginPath();
    s.data.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))));
    g.stroke();
  }

  if (cursor !== null && cursor >= 0 && cursor < xs.length) {
    const x = X(cursor);
    g.strokeStyle = CSS('--text-muted'); g.lineWidth = 1;
    g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(x, padT); g.lineTo(x, padT + plotH); g.stroke();
    g.setLineDash([]);
    series.forEach((s, n) => {
      const v = s.data[cursor];
      g.fillStyle = s.color;
      g.beginPath(); g.arc(x, Y(v), 4, 0, Math.PI * 2); g.fill();
      g.strokeStyle = CSS('--surface-1'); g.lineWidth = 2; g.stroke();
      g.fillStyle = CSS('--text-primary');
      g.font = '600 11px ui-monospace, monospace';
      g.textAlign = x > w * 0.6 ? 'right' : 'left';
      g.fillText(v.toFixed(1), x + (x > w * 0.6 ? -8 : 8), padT + 11 + n * 13);
    });
  }
}

function drawCharts() {
  if (!S.sweep) return;
  const x = S.sweep.theta_deg;
  drawChart($('chart-pressure'), x,
    [{ data: S.sweep.pressure_bar, color: CSS('--series-1') }], S.idx);
  drawChart($('chart-force'), x, [
    { data: S.sweep.f_pin_kn, color: CSS('--series-2') },
    { data: S.sweep.f_side_kn, color: CSS('--series-4') },
  ], S.idx);
  drawChart($('chart-torque'), x,
    [{ data: S.sweep.torque_nm, color: CSS('--series-3') }], S.idx);
}

/* -------------------------------------------------------------------- 3D */

let renderer, scene, camera, controls;

const MATERIALS = {
  piston: { color: 0xb9bcc0, metalness: 0.55, roughness: 0.42 },
  pin: { color: 0x8d9299, metalness: 0.85, roughness: 0.25 },
  rod: { color: 0x9aa0a8, metalness: 0.8, roughness: 0.3 },
  // The sleeve is a shell you look through, so it writes no depth -- with
  // depthWrite on it hides everything inside it however low the opacity.
  sleeve: { color: 0x7d838c, metalness: 0.3, roughness: 0.7,
            transparent: true, opacity: 0.13, side: THREE.DoubleSide,
            depthWrite: false },
};

function initScene() {
  const host = $('viewport');
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x0d0d0d);
  renderer.localClippingEnabled = true;
  host.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(36, 1, 1, 6000);
  camera.position.set(320, 150, 330);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.HemisphereLight(0xdfe6f2, 0x101014, 0.85));
  const key = new THREE.DirectionalLight(0xffffff, 0.85);
  key.position.set(220, 320, 240); scene.add(key);
  const fill = new THREE.DirectionalLight(0x93b4e8, 0.35);
  fill.position.set(-260, 120, -180); scene.add(fill);

  // Model +Z is the cylinder axis; rotate so it points up on screen.
  S.group = new THREE.Group();
  S.group.rotation.x = -Math.PI / 2;
  scene.add(S.group);

  // Section plane: normal along the pin axis, so the cut reveals the
  // under-crown cavity, the bosses and the rod.
  S.clip = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);

  resize();
  window.addEventListener('resize', () => { resize(); drawCharts(); });
  animate();
}

function resize() {
  const host = $('viewport');
  const w = host.clientWidth, h = host.clientHeight;
  if (!w || !h) return;
  // updateStyle must stay on. With it off the canvas gets width/height
  // ATTRIBUTES but no CSS size, so its intrinsic pixel size becomes its
  // layout size -- on a 150% display that is 1.5x too wide, and it pushes
  // the right rail off the screen.
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function meshFromTessellation(part, data) {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(data.vertices.length * 3);
  data.vertices.forEach((v, i) => {
    positions[i * 3] = v[0]; positions[i * 3 + 1] = v[1]; positions[i * 3 + 2] = v[2];
  });
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(data.triangles.flat());
  geometry.computeVertexNormals();
  return new THREE.Mesh(geometry, new THREE.MeshStandardMaterial(MATERIALS[part]));
}

async function loadGeometry() {
  note('building solids...');
  const g = await api('/api/geometry?tolerance=0.4');
  if (!g.ok) { toast(`<strong>Geometry refused:</strong> ${g.error}`, true); note(''); return; }

  // A stress field belongs to the design it was solved on. Rebuilding the
  // geometry means that design is gone, so the field goes with it rather
  // than sitting there describing a part that no longer exists.
  clearFea();
  Object.values(S.meshes).forEach((m) => S.group.remove(m));
  S.meshes = {};
  for (const [part, data] of Object.entries(g.meshes)) {
    const mesh = meshFromTessellation(part, data);
    S.meshes[part] = mesh;
    S.group.add(mesh);
  }

  if (!S.arrows.pin) {
    const mk = (hex) => {
      const a = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(), 40, hex, 12, 7);
      S.group.add(a); return a;
    };
    S.arrows.pin = mk(0xd95926);
    S.arrows.side = mk(0x9085e9);
  }
  note('');
  applyViewMode();
  frameAssembly();
}

/* Fit the camera to whatever was just built. Without this the camera sits on
 * the crank axis at the origin while the assembly stands a couple of hundred
 * millimetres above it, and the first thing you see is the inside of the
 * sleeve. */
function frameAssembly() {
  const box = new THREE.Box3();
  for (const [part, mesh] of Object.entries(S.meshes)) {
    if (part === 'sleeve') continue;            // the sleeve is the tall one
    box.expandByObject(mesh);
  }
  if (box.isEmpty()) return;

  const centre = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.72;
  const distance = radius / Math.sin((camera.fov * Math.PI / 180) / 2);

  controls.target.copy(centre);
  camera.position.set(centre.x + distance * 0.62,
                      centre.y + distance * 0.34,
                      centre.z + distance * 0.70);
  camera.near = Math.max(distance / 100, 0.5);
  camera.far = distance * 12;
  camera.updateProjectionMatrix();
  controls.update();

  // Keep the lights on the part rather than on the crank axis.
  scene.children.filter((c) => c.isDirectionalLight).forEach((light) => {
    light.target.position.copy(centre);
    scene.add(light.target);
  });
}

function applyViewMode() {
  const planes = S.mode === 'section' ? [S.clip] : [];
  for (const [part, mesh] of Object.entries(S.meshes)) {
    mesh.material.clippingPlanes = planes;
    mesh.material.needsUpdate = true;
    mesh.visible = (part !== 'sleeve') || S.showSleeve;
  }
  Object.values(S.arrows).forEach((a) => { a.visible = S.showForces; });
  placeParts();
}

function placeParts() {
  if (!S.sweep || !S.meshes.piston) return;
  const i = S.idx, s = S.sweep;
  const burst = S.mode === 'exploded' ? 1 : 0;

  S.meshes.piston.position.set(0, 0, s.piston_translate_mm[i] + burst * 70);
  S.meshes.pin.position.set(0, burst * 90, s.pin_z_mm[i]);
  S.meshes.rod.position.set(0, 0, s.pin_z_mm[i] - burst * 35);
  S.meshes.rod.rotation.y = s.rod_rotation_y[i];
  if (S.meshes.sleeve) S.meshes.sleeve.position.set(0, 0, s.sleeve_translate_mm);

  if (S.showForces) {
    const pin = s.f_pin_kn[i], side = s.f_side_kn[i];
    const scale = 60 / Math.max(s.peak_f_pin_kn, 1e-6);
    const sideScale = 45 / Math.max(s.peak_f_side_kn, 1e-6);
    const z = s.pin_z_mm[i];

    S.arrows.pin.position.set(0, 0, z);
    S.arrows.pin.setDirection(new THREE.Vector3(0, 0, pin >= 0 ? -1 : 1));
    S.arrows.pin.setLength(Math.max(Math.abs(pin) * scale, 1), 11, 6);

    S.arrows.side.position.set(0, 0, z);
    S.arrows.side.setDirection(new THREE.Vector3(side >= 0 ? 1 : -1, 0, 0));
    S.arrows.side.setLength(Math.max(Math.abs(side) * sideScale, 1), 9, 5);
  }
}

function animate() {
  requestAnimationFrame(animate);
  if (S.playing && S.sweep) {
    S.idx = (S.idx + 1) % S.sweep.theta_deg.length;
    setCrankIndex(S.idx, true);
  }
  controls.update();
  renderer.render(scene, camera);
}

function setCrankIndex(i, fromPlayback) {
  S.idx = i;
  const deg = S.sweep ? S.sweep.theta_deg[i] : 0;
  $('crank-value').textContent = `${deg.toFixed(0)}°`;
  if (!fromPlayback) placeParts(); else placeParts();
  if (!fromPlayback || i % 3 === 0) drawCharts();
  if (fromPlayback) $('crank').value = String(i);
  // The stress field follows the slider. One solve covers the whole cycle
  // because the analysis is linear, so this is a multiply per vertex.
  if (S.fea && S.fea.data.cycle) {
    paintStress(S.fea.mode, S.fea.data.cycle.scale[i]);
  }
}

/* ------------------------------------------------------------- assistant */

const CHAT = { busy: false, node: null, ready: null };

function chatEl(cls, html) {
  const el = document.createElement('div');
  el.className = cls;
  if (html !== undefined) el.innerHTML = html;
  $('chat-log').appendChild(el);
  $('chat-log').scrollTop = $('chat-log').scrollHeight;
  return el;
}

const escape = (t) => String(t).replace(/[&<>]/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

function renderProposal(ev) {
  const changes = Object.entries(ev.changes).map(([path, v]) => {
    const p = findParam(path);
    const shown = (p && typeof v === 'number')
      ? `${fmt(toDisplay(v, p.unit), 4)} ${unitLabel(p.unit)}` : v;
    const from = (p && typeof p.value === 'number')
      ? `${fmt(toDisplay(p.value, p.unit), 4)} \u2192 ` : '';
    return `<div class="change">${path}: ${from}${shown}</div>`;
  }).join('');

  const worse = (ev.regressions || []).slice(0, 4).map((r) =>
    `${r.metric.split('.').pop().replace(/_/g, ' ')} ${r.percent > 0 ? '+' : ''}${fmt(r.percent, 1)}%`);

  const card = chatEl('proposal', `
    <h4>Proposed change</h4>
    ${changes}
    ${ev.rationale ? `<div class="why">${escape(ev.rationale)}</div>` : ''}
    ${worse.length ? `<div class="worse">Also got worse: ${worse.join(', ')}</div>`
                   : '<div class="worse" style="color:var(--good)">Nothing tracked got worse.</div>'}
    <div class="actions">
      <button class="accept">Accept</button>
      <button class="discard">Discard</button>
    </div>`);

  const settle = async (endpoint, label) => {
    await post(`${endpoint}/${ev.proposal_id}`);
    card.classList.add('settled');
    card.querySelector('.actions').innerHTML =
      `<span style="font-size:11px;color:var(--text-muted)">${label}</span>`;
    await refresh();
  };
  card.querySelector('.accept').onclick = () => settle('/api/commit', 'Accepted');
  card.querySelector('.discard').onclick = () => settle('/api/discard', 'Discarded');
}

function findParam(path) {
  if (!S.state) return null;
  for (const group of S.state.sections) {
    for (const p of group.parameters) if (p.path === path) return p;
  }
  return null;
}

async function sendChat(text) {
  if (CHAT.busy || !text.trim()) return;
  CHAT.busy = true;
  $('chat-send').disabled = true;
  $('chat-text').value = '';
  chatEl('msg user', escape(text));
  CHAT.node = null;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok) throw new Error(`server returned ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop();
      for (const part of parts) {
        if (!part.startsWith('data: ')) continue;
        handleChatEvent(JSON.parse(part.slice(6)));
      }
    }
  } catch (e) {
    chatEl('msg failed', `Could not reach the assistant: ${escape(e.message)}`);
  } finally {
    CHAT.busy = false;
    $('chat-send').disabled = false;
  }
}

function handleChatEvent(ev) {
  switch (ev.type) {
    case 'text':
      if (!CHAT.node) CHAT.node = chatEl('msg assistant', '');
      CHAT.node.textContent += ev.text;
      $('chat-log').scrollTop = $('chat-log').scrollHeight;
      break;
    case 'tool_use': {
      CHAT.node = null;
      const detail = Object.keys(ev.input || {}).length
        ? ` ${escape(JSON.stringify(ev.input))}` : '';
      chatEl('chip',
        `<span class="dot"></span><span class="what">${ev.name}${detail}</span>`);
      break;
    }
    case 'tool_result':
      if (!ev.ok) {
        const chips = document.querySelectorAll('#chat-log .chip');
        if (chips.length) chips[chips.length - 1].classList.add('failed');
      }
      break;
    case 'proposal':
      CHAT.node = null;
      renderProposal(ev);
      break;
    case 'design_changed':
      refresh();
      break;
    case 'error':
      CHAT.node = null;
      chatEl('msg failed', escape(ev.message));
      break;
    default:
      break;
  }
}

async function initChat() {
  const status = await api('/api/ai/status');
  CHAT.ready = status.available;
  if (!status.available) {
    const note = document.createElement('div');
    note.className = 'chat-note';
    note.textContent = status.reason;
    $('tab-assistant').insertBefore(note, $('tab-assistant').querySelector('.chat-input'));
    $('chat-text').placeholder = 'Assistant unavailable \u2014 no API key';
  }
  $('chat-send').addEventListener('click', () => sendChat($('chat-text').value));
  $('chat-text').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChat($('chat-text').value);
    }
  });
  $('chat-log').addEventListener('click', (e) => {
    if (e.target.classList.contains('example')) sendChat(e.target.textContent);
  });
  $('rail-tabs').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    [...e.currentTarget.children].forEach((c) => c.classList.toggle('on', c === b));
    $('tab-margins').hidden = b.dataset.tab !== 'margins';
    $('tab-optimise').hidden = b.dataset.tab !== 'optimise';
    $('tab-assistant').hidden = b.dataset.tab !== 'assistant';
    if (b.dataset.tab === 'assistant') $('chat-text').focus();
  });
  $('opt-run').addEventListener('click', runOptimise);
  $('opt-front').addEventListener('click', runFrontier);
}

/* ------------------------------------------------------------------ boot */


/* --- block envelope ----------------------------------------------------
 * The one constraint that comes from the block being a real object rather
 * than a design choice. It sits above the parameters because it should be
 * read BEFORE someone types a bigger bore, not after the tool refuses one.
 *
 * The tier badge is the point of the panel as much as the number is. A
 * ceiling computed from an assumed wall thickness and a ceiling taken from
 * what pistons are actually sold are not the same kind of fact, and showing
 * them identically would be the tool lying by presentation.
 */
const TIER_LABEL = { 1: 'tier 1 measured', 2: 'tier 2 catalogue',
                     3: 'tier 3 ASSUMED' };

const RESLEEVE_NOTE =
  '<br><br>Resleeving is ON: the bore may also go below as-built.';

function renderEnvelope(env) {
  const host = $('envelope');
  if (!env || !env.ok) { host.hidden = true; return; }
  host.hidden = false;

  const mm = (v) => (v * 1000);
  if (env.max_bore_m === null || env.max_bore_m === undefined) {
    host.innerHTML = `<h3>Block envelope</h3>`
      + `<p class="why">No limit could be evaluated, so the bore is pinned `
      + `where it is. Supply the bore spacing, the liner thickness or a `
      + `catalogue bore to open it up.</p>`;
    return;
  }

  const used = mm(env.current_bore_m) - mm(env.as_built_bore_m);
  const span = mm(env.max_bore_m) - mm(env.as_built_bore_m);
  const pct = span > 0 ? Math.max(0, Math.min(100, used / span * 100)) : 100;
  const tier = env.confidence_tier;
  const unknown = (env.unknown || []).length;

  host.innerHTML = `
    <h3>Block envelope</h3>
    <div class="row"><span>as built</span>
      <b>${mm(env.as_built_bore_m).toFixed(3)} mm</b></div>
    <div class="row"><span>now</span>
      <b>${mm(env.current_bore_m).toFixed(3)} mm</b></div>
    <div class="bar"><span style="width:${pct}%"></span></div>
    <div class="row"><span>${mm(env.headroom_m).toFixed(3)} mm left</span>
      <b>max ${mm(env.max_bore_m).toFixed(3)} mm</b></div>
    <p class="why">
      <span class="tier tier-${tier}">${TIER_LABEL[tier] || 'tier ?'}</span>
      &nbsp;set by ${env.binding.name}.<br>${env.binding.reason}
      ${unknown ? `<br><br><b>${unknown} limit${unknown > 1 ? 's' : ''}
        could not be evaluated</b> (${(env.unknown || []).join(', ')}), so
        this ceiling may be optimistic.` : ''}
      ${env.resleeving ? RESLEEVE_NOTE : ''}
    </p>`;
}


/* --- optimiser ---------------------------------------------------------
 * Nothing this panel produces is committed. A searched design is a proposal
 * like any other, and the warnings are shown ABOVE the numbers on purpose:
 * an optimum that sits on a calibrated guess is worth what the guess is, and
 * burying that under a torque figure would be the most misleading thing this
 * tool could do.
 */

function drawFrontier(canvas, points, labels) {
  const dpr = window.devicePixelRatio || 1;
  const h = 170;
  canvas.style.width = '100%';
  canvas.style.height = h + 'px';
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  const usable = points.filter((p) => p.feasible);
  if (usable.length < 2) {
    g.fillStyle = CSS('--text-muted');
    g.font = '12px system-ui';
    g.textAlign = 'center';
    g.fillText('not enough feasible points to draw a curve', w / 2, h / 2);
    return;
  }

  const padL = 44, padR = 12, padT = 12, padB = 30;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const xs = usable.map((p) => p.level);
  const ys = usable.map((p) => p.value);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y1 === y0) y1 = y0 + 1;
  const yPad = (y1 - y0) * 0.12; y0 -= yPad; y1 += yPad;

  const X = (v) => padL + ((v - x0) / (x1 - x0 || 1)) * plotW;
  const Y = (v) => padT + (1 - (v - y0) / (y1 - y0)) * plotH;

  g.strokeStyle = CSS('--grid'); g.lineWidth = 1;
  g.fillStyle = CSS('--text-muted');
  g.font = '10px ui-monospace, monospace';
  for (let k = 0; k <= 3; k += 1) {
    const v = y0 + (y1 - y0) * (k / 3);
    const y = Math.round(Y(v)) + 0.5;
    g.beginPath(); g.moveTo(padL, y); g.lineTo(w - padR, y); g.stroke();
    g.textAlign = 'right';
    g.fillText(v.toFixed(Math.abs(v) < 10 ? 2 : 0), padL - 5, y + 3);
  }

  // Label the ends only. Five ticks in a 270 px rail overlap into mush.
  g.textAlign = 'left';
  g.fillText(usable[0].level.toFixed(2), padL, h - 16);
  g.textAlign = 'right';
  g.fillText(usable[usable.length - 1].level.toFixed(2), w - padR, h - 16);
  g.textAlign = 'center';
  g.fillText(labels.x, w / 2, h - 4);

  g.strokeStyle = CSS('--series-3') || '#7cb342';
  g.lineWidth = 2;
  g.beginPath();
  usable.forEach((p, i) => {
    const px = X(p.level), py = Y(p.value);
    if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
  });
  g.stroke();

  g.fillStyle = CSS('--series-3') || '#7cb342';
  for (const p of usable) {
    g.beginPath();
    g.arc(X(p.level), Y(p.value), 3.5, 0, Math.PI * 2);
    g.fill();
  }

  // The design you have now, marked so the curve reads as a price list
  // rather than a set of options with no origin.
  const here = usable[0];
  g.strokeStyle = CSS('--baseline') || 'rgba(255,255,255,0.35)';
  g.setLineDash([3, 3]); g.lineWidth = 1;
  g.beginPath(); g.moveTo(X(here.level), padT);
  g.lineTo(X(here.level), padT + plotH); g.stroke();
  g.setLineDash([]);
}

function renderOptimiseResult(r) {
  const host = $('opt-result');
  if (!r || !r.ok) {
    host.innerHTML = `<p class="doubt">${(r && r.error) || 'failed'}</p>`;
    return;
  }
  const pct = (v) => (v === null || v === undefined ? '' :
    `<span class="${v >= 0 ? 'up' : 'down'}">${v >= 0 ? '+' : ''}${v.toFixed(2)}%</span>`);

  const doubts = (r.warnings || [])
    .map((w) => `<p class="doubt">${w}</p>`).join('');

  const moves = (r.moves || []).map((m) => `
    <tr><td>${m.path}${m.at_bound ? ` <em>(at ${m.at_bound})</em>` : ''}</td>
        <td class="num">${pct(m.percent)}</td></tr>`).join('');

  const worse = (r.worse || []).map((row) => `
    <tr><td>${row.label}</td>
        <td class="num">${pct(row.percent)}</td></tr>`).join('');

  host.innerHTML = `
    ${doubts}
    <h4>${r.objective}</h4>
    <table>
      <tr><td>before</td><td class="num">${r.before.toPrecision(5)}</td></tr>
      <tr><td>after</td><td class="num">${r.after.toPrecision(5)}
        &nbsp;${pct(r.gain_percent)}</td></tr>
      <tr><td>safety factor</td>
        <td class="num">${r.safety_before.toFixed(3)} &rarr;
          ${r.safety_after.toFixed(3)}</td></tr>
      <tr><td>binding</td><td class="num">${r.binding_after}</td></tr>
    </table>
    ${moves ? `<h4>levers moved</h4><table>${moves}</table>` : ''}
    ${worse ? `<h4>what got worse</h4><table>${worse}</table>` : ''}
    <p style="margin-top:10px;color:#9aa0ab">${r.evaluations} evaluations.
      Nothing has been committed.</p>`;
}

async function runOptimise() {
  const objective = $('opt-objective').value;
  const buttons = [$('opt-run'), $('opt-front')];
  buttons.forEach((b) => { b.disabled = true; });
  $('opt-result').innerHTML = '<p>searching...</p>';
  try {
    const r = await api(`/api/optimise?objective=${objective}&levers=6`);
    renderOptimiseResult(r);
  } catch (e) {
    $('opt-result').innerHTML = `<p class="doubt">${e.message}</p>`;
  }
  buttons.forEach((b) => { b.disabled = false; });
}

async function runFrontier() {
  const objective = $('opt-objective').value;
  const buttons = [$('opt-run'), $('opt-front')];
  buttons.forEach((b) => { b.disabled = true; });
  $('opt-result').innerHTML = '<p>tracing the frontier...</p>';
  try {
    const r = await api(`/api/frontier?objective=${objective}&points=5&levers=5`);
    if (!r.ok) {
      $('opt-result').innerHTML = `<p class="doubt">${r.error}</p>`;
    } else {
      drawFrontier($('chart-frontier'), r.points,
        { x: 'minimum safety factor' });
      const rows = r.points.filter((p) => p.feasible);
      const base = rows.length ? rows[0].value : 0;
      $('opt-result').innerHTML = `
        <h4>${r.objective} against safety factor</h4>
        <table>${rows.map((p) => `
          <tr><td>SF ${p.level.toFixed(3)}</td>
              <td class="num">${p.value.toPrecision(5)}
                <span class="${p.value >= base ? 'up' : 'down'}">
                ${base ? ((p.value - base) / Math.abs(base) * 100).toFixed(2) : '0'}%
                </span></td></tr>
          <tr><td colspan="2" style="color:#9aa0ab;padding-bottom:6px">
              ${p.binding}</td></tr>`).join('')}</table>
        ${(r.notes || []).map((n) => `<p class="doubt">${n}</p>`).join('')}`;
    }
  } catch (e) {
    $('opt-result').innerHTML = `<p class="doubt">${e.message}</p>`;
  }
  buttons.forEach((b) => { b.disabled = false; });
}

async function refresh() {
  const [state, sweep, margins, metrics, envelope] = await Promise.all([
    api('/api/state'), api('/api/sweep'), api('/api/margins'),
    api('/api/metrics'), api('/api/envelope').catch(() => null),
  ]);
  S.state = state; S.sweep = sweep; S.margins = margins; S.metrics = metrics;
  S.envelope = envelope;

  // A stress field describes one design. Any change to the geometry or the
  // loads makes it a picture of a part that no longer exists, and a stale
  // picture is worse than none: it looks exactly like a current one.
  if (S.fea && S.fea.fingerprint && state.fingerprint !== S.fea.fingerprint) {
    const part = S.fea.part;
    clearFea();
    toast(`The design changed, so the <strong>${part}</strong> stress field `
          + 'was dropped. Run it again to see the new geometry.');
  }

  renderEnvelope(envelope);
  renderRail(state);
  renderMargins(margins);
  $('rpm').value = Math.round(sweep.rpm);
  $('crank').max = String(sweep.theta_deg.length - 1);
  if (S.idx >= sweep.theta_deg.length || S.idx === 0) {
    // Open at firing TDC: the interesting part of the cycle, not its edge.
    S.idx = sweep.theta_deg.findIndex((d) => d >= 0);
    if (S.idx < 0) S.idx = 0;
  }
  $('crank').value = String(S.idx);

  drawCharts();
  placeParts();
  note('');

  const failing = margins.failing_count;
  if (failing) {
    toast(`<strong>${failing} margin${failing > 1 ? 's' : ''} below 1.0.</strong> `
      + `Binding: ${margins.binding}`, true);
  }
}


/* --- stress field ------------------------------------------------------
 * The FEA mesh is its own geometry, not a recolouring of the display mesh:
 * it has different vertices, because tetgen inserts its own. So running a
 * case swaps the part's mesh out and restoring puts the original back.
 *
 * Colour is a per-vertex attribute rather than a texture. Vertex colours on
 * a MeshStandardMaterial still take the scene lighting, which keeps the part
 * readable as a shape instead of flattening it into a heat map.
 */

// Green through yellow to red. Matched by .legend-bar in styles.css --
// change one and change the other.
const STRESS_RAMP = [
  [0.00, 0x1f, 0x9d, 0x55],
  [0.28, 0x7c, 0xb3, 0x42],
  [0.55, 0xf2, 0xc2, 0x00],
  [0.78, 0xef, 0x7d, 0x1a],
  [1.00, 0xd4, 0x3f, 0x2a],
];

function rampColour(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STRESS_RAMP.length; i += 1) {
    const [p1, r1, g1, b1] = STRESS_RAMP[i];
    if (t <= p1 || i === STRESS_RAMP.length - 1) {
      const [p0, r0, g0, b0] = STRESS_RAMP[i - 1];
      const f = p1 === p0 ? 0 : (t - p0) / (p1 - p0);
      return [(r0 + (r1 - r0) * f) / 255,
              (g0 + (g1 - g0) * f) / 255,
              (b0 + (b1 - b0) * f) / 255];
    }
  }
  return [1, 1, 1];
}

function meshFromField(field) {
  const geometry = new THREE.BufferGeometry();
  const n = field.vertices.length;
  const positions = new Float32Array(n * 3);
  for (let i = 0; i < n; i += 1) {
    const v = field.vertices[i];
    positions[i * 3] = v[0];
    positions[i * 3 + 1] = v[1];
    positions[i * 3 + 2] = v[2];
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color',
    new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  geometry.setIndex(field.triangles.flat());
  geometry.computeVertexNormals();
  return new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
    vertexColors: true, metalness: 0.15, roughness: 0.62,
  }));
}

/* Two ways to read the same field, and they answer different questions.
 *
 *   distribution  scaled to this field's own 99th percentile. Shows WHERE
 *                 the load concentrates. Says nothing about whether the
 *                 numbers are large, which is why it is easy to mistake a
 *                 red patch for a part about to let go.
 *
 *   vs yield      scaled to the material's allowable at the temperature this
 *                 part actually runs at. Answers "how close are we", and on
 *                 a healthy design most of the part should sit at the cold
 *                 end of the ramp.
 *
 * Stress scales exactly with load in a linear analysis, so recolouring for a
 * different crank angle is a multiply -- no re-solve, no network round trip.
 */
function paintStress(mode, scale) {
  if (!S.fea) return;
  const { mesh, data } = S.fea;
  const field = data.field;
  const base = field.stress_mpa;
  const colours = mesh.geometry.getAttribute('color');
  const factor = scale === undefined ? S.fea.scale : scale;

  const allowableMpa = data.allowable
    ? data.allowable.allowable_pa / 1e6 : null;

  let lo = 0;
  let hi;
  if (mode === 'yield' && allowableMpa) {
    hi = allowableMpa;
  } else {
    lo = field.range_mpa.min * factor;
    hi = (field.range_mpa.clip || field.range_mpa.max) * factor;
  }
  const span = Math.max(hi - lo, 1e-9);

  for (let i = 0; i < base.length; i += 1) {
    const [r, g, b] = rampColour((base[i] * factor - lo) / span);
    colours.array[i * 3] = r;
    colours.array[i * 3 + 1] = g;
    colours.array[i * 3 + 2] = b;
  }
  colours.needsUpdate = true;
  S.fea.scale = factor;
  S.fea.mode = mode;
  updateFeaLegend();
}

function updateFeaLegend() {
  if (!S.fea) return;
  const { data, mode, scale } = S.fea;
  const field = data.field;
  if (!data.allowable) return;
  const allowable = data.allowable.allowable_pa / 1e6;
  const peak = field.range_mpa.max * scale;

  if (mode === 'yield') {
    $('legend-lo').textContent = '0';
    $('legend-hi').textContent = `${allowable.toFixed(0)} MPa`;
  } else {
    $('legend-lo').textContent =
      `${(field.range_mpa.min * scale).toFixed(0)} MPa`;
    $('legend-hi').textContent =
      `${((field.range_mpa.clip || field.range_mpa.max) * scale).toFixed(0)}+`;
  }

  const share = peak / allowable * 100;
  const angle = S.sweep && data.cycle
    ? `${S.sweep.theta_deg[S.idx].toFixed(0)}\u00b0` : 'the governing angle';
  const loadNow = data.cycle
    ? `${(data.cycle.load[S.idx] / (data.cycle.unit === 'Pa' ? 1e6 : 1e3))
        .toFixed(1)} ${data.cycle.unit === 'Pa' ? 'MPa' : 'kN'}`
    : '';

  // The solver says useful things -- what load it actually applied, how many
  // slivers it threw away -- and until now every one of them was discarded
  // before it reached anybody. They are the difference between "the rod is
  // green" and "the rod is green BECAUSE the load was 0.2 kN at this angle".
  const notes = (data.notes || [])
    .map((n) => `<li>${n}</li>`).join('');

  $('fea-caption').innerHTML =
    `${field.element_count.toLocaleString()} elements &middot; `
    + `${angle}${loadNow ? ` &middot; ${loadNow}` : ''}<br>`
    + `peak ${peak.toFixed(0)} MPa &middot; `
    + `<b>${share.toFixed(0)}% of ${data.allowable.basis.split(' ')[0]}</b> `
    + `(${allowable.toFixed(0)} MPa at ${data.allowable.temperature_c.toFixed(0)}&deg;C)<br>`
    + `<em>that peak is one element under a rigid restraint, not a `
    + `material stress \u2014 read the body of the part, not the hot spot</em>`
    + (notes ? `<details class="fea-notes"><summary>what the solver `
        + `did</summary><ul>${notes}</ul></details>` : '');
}

// The buttons are built from what the server actually has a load case for,
// so adding a case in cases.py puts a button here with no front-end change.
async function initFeaButtons() {
  const host = $('fea-parts');
  let parts = [];
  try {
    const r = await api('/api/fea/cases');
    parts = (r && r.parts) || [];
  } catch (e) { parts = []; }
  if (!parts.length) { host.hidden = true; return; }
  host.hidden = false;
  host.innerHTML = `<span class="segmented-label">Stress</span>`
    + parts.map((p) => `<button data-fea="${p}">${p}</button>`).join('');
  host.addEventListener('click', (e) => {
    const part = e.target.dataset && e.target.dataset.fea;
    if (part) runFea(part);
  });
}

function markFeaButton(part) {
  $('fea-parts').querySelectorAll('button').forEach((b) => {
    b.classList.toggle('on', b.dataset.fea === part);
  });
}

async function runFea(part) {
  if (S.fea && S.fea.part === part) { clearFea(); return; }
  const buttons = $('fea-parts').querySelectorAll('button');
  buttons.forEach((b) => { b.disabled = true; });
  note(`meshing and solving the ${part}...`);
  let r;
  try {
    r = await api(`/api/fea/${part}?elements=25000`);
  } catch (e) {
    buttons.forEach((b) => { b.disabled = false; }); note('');
    toast(`<strong>Solve failed:</strong> ${e.message}`, true);
    return;
  }
  buttons.forEach((b) => { b.disabled = false; });
  note('');
  if (!r.ok) { toast(`<strong>Solve refused:</strong> ${r.error}`, true); return; }

  // A page newer than the server it is talking to is the single most
  // confusing state this tool can be in: the colours never change with the
  // crank angle, the yield scale does nothing, and any bug fixed since the
  // server started is still there -- three unrelated-looking symptoms with
  // one cause. Say so instead of failing three different quiet ways.
  const missing = [];
  if (!r.cycle) missing.push('crank-angle scaling');
  if (!r.allowable) missing.push('the yield reference');
  if (missing.length) {
    toast('<strong>The server is older than this page.</strong> It did not '
      + `send ${missing.join(' or ')}. Stop run.bat, start it again, and `
      + 'reload. (Open /api/version — if it 404s, that is the old server.)',
      true);
    return;
  }

  clearFea();
  const mesh = meshFromField(r.field);
  const original = S.meshes[part];
  if (original) {
    mesh.position.copy(original.position);
    mesh.rotation.copy(original.rotation);
    S.group.remove(original);
  }
  S.group.add(mesh);
  S.fea = {
    part, mesh, original, data: r,
    mode: $('fea-mode') && $('fea-mode').value === 'yield'
      ? 'yield' : 'distribution',
    // The solve was done at the governing angle; the slider may be elsewhere.
    scale: r.cycle ? r.cycle.scale[S.idx] : 1,
    fingerprint: S.state ? S.state.fingerprint : null,
  };
  S.meshes[part] = mesh;

  markFeaButton(part);
  // Let the part being analysed be seen. A pin solved in place is otherwise
  // entirely inside the piston, and the stress field is invisible.
  setAssemblyGhosted(true, part);
  paintStress(S.fea.mode, S.fea.scale);
  $('fea-legend').hidden = false;
}

/* Push everything except `keep` back to a ghost, so the analysed part reads
 * against the assembly instead of being buried in it. The originals' own
 * materials are stashed rather than edited, because the sleeve already has a
 * deliberate transparency of its own that must come back intact. */
function setAssemblyGhosted(on, keep) {
  for (const [part, mesh] of Object.entries(S.meshes)) {
    if (part === keep || !mesh.material) continue;
    if (on) {
      if (!mesh.userData.solidMaterial) {
        mesh.userData.solidMaterial = mesh.material;
        mesh.material = new THREE.MeshStandardMaterial({
          color: 0x8a9099, metalness: 0.1, roughness: 0.9,
          transparent: true, opacity: 0.07,
          depthWrite: false, side: THREE.DoubleSide,
        });
      }
    } else if (mesh.userData.solidMaterial) {
      mesh.material.dispose();
      mesh.material = mesh.userData.solidMaterial;
      delete mesh.userData.solidMaterial;
    }
  }
}

function clearFea() {
  if (!S.fea) return;
  setAssemblyGhosted(false);
  S.group.remove(S.fea.mesh);
  S.fea.mesh.geometry.dispose();
  S.fea.mesh.material.dispose();
  if (S.fea.original) {
    S.group.add(S.fea.original);
    S.meshes[S.fea.part] = S.fea.original;
  }
  S.fea = null;
  markFeaButton(null);
  $('fea-legend').hidden = true;
}

function wire() {
  $('crank').addEventListener('input', (e) => {
    S.playing = false; $('play').textContent = 'Play';
    setCrankIndex(Number(e.target.value), false);
  });
  $('play').addEventListener('click', () => {
    S.playing = !S.playing;
    $('play').textContent = S.playing ? 'Pause' : 'Play';
  });
  $('viewmode').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    [...e.currentTarget.children].forEach((c) => c.classList.toggle('on', c === b));
    S.mode = b.dataset.mode;
    applyViewMode();
  });
  $('show-forces').addEventListener('change', (e) => {
    S.showForces = e.target.checked; applyViewMode();
  });
  $('show-sleeve').addEventListener('change', (e) => {
    S.showSleeve = e.target.checked; applyViewMode();
  });
  $('rpm').addEventListener('change', async (e) => {
    note('evaluating...');
    await post(`/api/rpm/${Number(e.target.value)}`);
    await refresh();
  });
  $('undo').addEventListener('click', async () => {
    await post('/api/revert'); await refresh();
  });
  $('save').addEventListener('click', async () => {
    const r = await post('/api/save');
    toast(r.ok ? `Saved to ${r.path}` : `Not saved: ${r.error}`, !r.ok);
  });
  $('fea-clear').addEventListener('click', clearFea);
  $('fea-mode').addEventListener('change', (e) => {
    if (S.fea) paintStress(e.target.value);
  });
}

(async function boot() {
  initScene();
  wire();
  await initFeaButtons();
  await initChat();
  await refresh();
  await loadGeometry();
  setCrankIndex(S.idx, false);
})().catch((e) => {
  note('failed to start');
  toast(`<strong>Startup failed:</strong> ${e.message}`, true);
});
