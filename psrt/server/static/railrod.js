/* Rail connecting rod -- the right-rail panel, the 2D section and the
 * per-part FEA display.
 *
 * Loaded after app.js and uses its globals (S, api, post, write, toast,
 * note, meshFromField, rampColour, setAssemblyGhosted, clearFea). Nothing in
 * app.js depends on this file beyond three optional hooks: window.onRefresh,
 * window.onCrankChange and window.showRailRod.
 */
'use strict';

const RR = {
  data: null, section: null, zoom: false, swing: 0, forces: 'assembly',
  timer: null, feaTimer: null, visible: false,
};

const RR_COLOURS = {
  rails: '#a7adb5', sleeve: '#3f7f5f', receiver: '#c49a6c',
  clamp_right: '#6f8fd1', clamp_left: '#8b76d1', bolt: '#d06a5a',
  crankpin: '#2b2b2a', eye_bore: '#121211',
};
const RR_FORCE_COLOURS = {
  floor: '#f2c200', web: '#f2c200', flank: '#d95926', knob: '#ef7d1a',
  guide: '#3987e5', clamp_pin: '#199e70', receiver_pin: '#199e70',
  lug: '#9085e9',
};
const RR_PART_NAMES = {
  rr_rails: 'rails', rr_sleeve: 'sleeve', rr_receiver: 'receiver',
  rr_clamp_right: 'clamp R', rr_clamp_left: 'clamp L', rr_bolt: 'bolt',
};

const kN = (n) => `${(n / 1e3).toFixed(1)} kN`;

window.showRailRod = async function showRailRod() {
  RR.visible = true;
  await loadRailRod();
};

window.onRefresh = function onRefresh() {
  if (!RR.visible) return;
  clearTimeout(RR.timer);
  RR.timer = setTimeout(loadRailRod, 150);
};

window.onCrankChange = function onCrankChange(i, fromPlayback) {
  if (RR.visible && RR.forces === 'crank' && !fromPlayback) {
    clearTimeout(RR.timer);
    RR.timer = setTimeout(loadSection, 120);
  }
  if (S.fea && S.fea.rail && S.fea.mode === 'angle') {
    clearTimeout(RR.feaTimer);
    RR.feaTimer = setTimeout(() => paintRailAngle(), fromPlayback ? 0 : 60);
  }
};

async function loadRailRod() {
  let r;
  try { r = await api('/api/railrod'); } catch (e) { return; }
  RR.data = r;
  const on = r.enabled;
  $('rr-enable').textContent = on ? 'Back to the conventional rod'
    : 'Use the rail rod';
  $('rr-state').textContent = on ? 'rail rod active'
    : 'the conventional I-beam rod is active';
  $('rr-body').hidden = !on;
  if (!on) return;
  if (!r.ok) {
    $('rr-cards').innerHTML = `<p class="rr-muted">${r.error}</p>`;
    return;
  }
  renderCards(r);
  renderMarginsRR(r.margins);
  renderFeaButtons();
  $('rr-notes').innerHTML = (r.summary.notes || [])
    .map((n) => `<p>&bull; ${n}</p>`).join('');
  await loadSection();
}

function card(k, v, s, cls) {
  return `<div class="rr-card ${cls || ''}"><div class="k">${k}</div>`
    + `<div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ''}</div>`;
}

function renderCards(r) {
  const s = r.coupled || r.summary;
  const base = r.summary;
  const total = base.total_mass_kg + r.shell_mass_kg;
  const conv = r.rod_total_specified_kg;
  const saving = (conv - total) / conv * 100;
  const w = base.swing;
  const ss = base.self_seating;
  const sep = s.separation_factor;
  $('rr-cards').innerHTML = [
    card('rod mass', `${(total * 1e3).toFixed(0)} g`,
      `incl. ${(r.shell_mass_kg * 1e3).toFixed(0)} g shells &middot; `
      + `${saving >= 0 ? '' : '+'}${(-saving).toFixed(0)}% vs the ${
        (conv * 1e3).toFixed(0)} g in the design state`,
      saving > 0 ? 'ok' : 'bad'),
    card('knob & flank fit', `${(base.conformity.gap_max_mm * 1e3).toFixed(0)} &micro;m`,
      base.conformity.conforms ? 'conforms to the notch' : 'DOES NOT CONFORM',
      base.conformity.conforms ? 'ok' : 'bad'),
    card('clamp swing', `${w.free_opening_deg.toFixed(0)}&deg; free`,
      w.hook_in_angle_deg !== null ? `hooks in at ${w.hook_in_angle_deg}&deg;`
        : 'cannot be hooked in', w.assemblable ? 'ok' : 'bad'),
    card('self-seating ramp', `${ss.ramp_angle_deg.toFixed(0)}&deg; vs ${
      ss.angle_for_self_seating_deg.toFixed(0)}&deg;`,
      ss.self_seating ? 'preload pushes the tip in' : 'preload swings the tip out',
      ss.self_seating ? 'ok' : 'bad'),
    card('rail seat preload', kN(s.assembly.rail_seat_preload_n),
      `${r.coupled ? 'coupled' : 'rigid'} network, at assembly`,
      s.assembly.rail_seat_preload_n > 0 ? 'ok' : 'bad'),
    card('seat opens at', sep === null ? 'never' : `${sep.toFixed(2)}&times;`,
      'multiple of peak tension', sep !== null && sep < 1 ? 'bad' : 'ok'),
    card('bolt', `${kN(s.bolt_range_n[0])} &ndash; ${kN(s.bolt_range_n[1])}`,
      `preload ${kN(s.assembly.bolt_n)}`,
      s.bolt_range_n[1] > 1.3 * s.assembly.bolt_n ? 'bad' : ''),
    card('receiver seat', kN(s.assembly.receiver_seat_n),
      'clamp on the receiver, at assembly'),
  ].join('');
  $('rr-network').textContent = r.coupled
    ? 'Contact forces from the FEA-coupled network.'
    : 'Contact forces from the RIGID network: fast, but optimistic about '
      + 'how preload is shared. Couple the FEA flexibility before believing '
      + 'the preload numbers.';
}

function renderMarginsRR(margins) {
  const bandOf = (sf) => (sf === null ? 'good' : sf < 1 ? 'critical'
    : sf < 1.25 ? 'serious' : sf < 1.5 ? 'warning' : 'good');
  $('rr-margins').innerHTML = margins.slice(0, 14).map((m) =>
    `<div class="rr-margin" title="${(m.equation || '').replace(/"/g, '&quot;')}">`
    + `<span>${m.component} &mdash; ${m.mode}</span>`
    + `<span class="sf" style="color:var(--${bandOf(m.safety_factor)})">${
      m.safety_factor === null ? 'inf' : m.safety_factor.toFixed(2)}</span></div>`)
    .join('');
}

async function loadSection() {
  const params = new URLSearchParams({ swing: String(RR.swing) });
  if (RR.forces === 'crank' && S.sweep) {
    params.set('theta_deg', String(S.sweep.theta_deg[S.idx]));
  }
  let r;
  try { r = await api(`/api/railrod/section?${params}`); } catch (e) { return; }
  if (!r.ok) return;
  RR.section = r;
  drawSection();
}

function drawSection() {
  const r = RR.section;
  if (!r) return;
  const svg = $('rr-svg');
  const flip = (p) => `${p[0].toFixed(3)},${(-p[1]).toFixed(3)}`;
  const ring = (pts) => `M${pts.map(flip).join('L')}Z`;
  let out = '';
  const order = ['crankpin', 'receiver', 'rails', 'sleeve', 'clamp_left',
    'clamp_right', 'bolt', 'eye_bore'];
  for (const part of order) {
    for (const poly of r.parts[part] || []) {
      const d = ring(poly.outer) + poly.holes.map(ring).join('');
      const fill = RR_COLOURS[part];
      const opacity = part === 'crankpin' ? 1 : part === 'sleeve' ? 0.5 : 0.85;
      out += `<path d="${d}" fill="${fill}" fill-opacity="${opacity}" `
        + `fill-rule="evenodd" stroke="#0d0d0d" stroke-width="0.08"/>`;
    }
  }
  // the knob pivot
  const pv = r.pivot;
  out += `<circle cx="${pv[0]}" cy="${-pv[1]}" r="0.25" fill="#fff"/>`
    + `<circle cx="${-pv[0]}" cy="${-pv[1]}" r="0.25" fill="#fff"/>`;

  if (RR.forces !== 'none' && r.contacts) {
    const fmax = Math.max(1, ...r.contacts.map((c) => c.force_n));
    const scale = (RR.zoom ? 3.5 : 9) / fmax;
    for (const c of r.contacts) {
      for (const side of [1, -1]) {
        const p = [side * c.point[0], c.point[1]];
        const n = [side * c.normal[0], c.normal[1]];
        const len = c.force_n * scale;
        // force on body b along +n, drawn arriving at the point
        const tail = [p[0] - n[0] * len, p[1] - n[1] * len];
        const col = RR_FORCE_COLOURS[c.group] || '#fff';
        out += `<line x1="${tail[0]}" y1="${-tail[1]}" x2="${p[0]}" `
          + `y2="${-p[1]}" stroke="${col}" stroke-width="${RR.zoom ? 0.08 : 0.3}" `
          + `marker-end="url(#rr-arrow)"/>`;
        if (Math.abs(c.point[0]) < 1e-6) break;
      }
    }
  }
  const view = RR.zoom
    ? `${pv[0] - 6} ${-pv[1] - 6} 12 12`
    : '-46 -46 92 92';
  svg.setAttribute('viewBox', view);
  svg.style.height = `${svg.clientWidth}px`;
  svg.innerHTML = '<defs><marker id="rr-arrow" viewBox="0 0 10 10" refX="9" '
    + 'refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">'
    + '<path d="M0,0L10,5L0,10z" fill="#ddd"/></marker></defs>' + out;
  const net = r.network === 'coupled' ? 'coupled' : 'rigid';
  $('rr-network').textContent = RR.forces === 'none' ? ''
    : `Arrows: contact forces at ${r.forces_at} (${net} network), bolt ${
      kN(r.bolt_n)}. Yellow floor/web, orange notch, blue receiver seat, `
      + 'green crankpin, violet lug faces.';
}

function renderFeaButtons() {
  const host = $('rr-fea');
  if (host.dataset.built) {
    markRailButtons();
    return;
  }
  host.dataset.built = '1';
  host.innerHTML = Object.entries(RR_PART_NAMES)
    .map(([p, label]) => `<button class="ghost" data-part="${p}">${label}</button>`)
    .join('');
  host.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (b) runRailFea(b.dataset.part);
  });
}

function markRailButtons() {
  $('rr-fea').querySelectorAll('button').forEach((b) => {
    b.classList.toggle('on', !!(S.fea && S.fea.rail && S.fea.part === b.dataset.part));
  });
}

async function runRailFea(part) {
  if (S.fea && S.fea.rail && S.fea.part === part) { clearFea(); markRailButtons(); return; }
  const buttons = $('rr-fea').querySelectorAll('button');
  buttons.forEach((b) => { b.disabled = true; });
  note(`solving the ${RR_PART_NAMES[part]} through the cycle...`);
  $('rr-fea-result').textContent = 'Meshing and solving. The first part on a '
    + 'new design also solves the rails, receiver and clamp for the coupled '
    + 'network, so allow a minute.';
  let r;
  try {
    r = await api(`/api/railrod/fea/${part}?elements=20000`);
  } catch (e) {
    r = { ok: false, error: e.message };
  }
  buttons.forEach((b) => { b.disabled = false; });
  note('');
  if (!r.ok) {
    toast(`<strong>Solve failed:</strong> ${r.error}`, true);
    $('rr-fea-result').textContent = r.error;
    return;
  }
  clearFea();
  const mesh = meshFromField({ vertices: r.field.vertices,
    triangles: r.field.triangles });
  const original = S.meshes[part];
  if (original) {
    mesh.position.copy(original.position);
    mesh.rotation.copy(original.rotation);
    S.group.remove(original);
  }
  S.group.add(mesh);
  S.fea = { part, mesh, original, data: r, rail: true, mode: 'angle',
    values: null, fingerprint: r.fingerprint };
  S.meshes[part] = mesh;
  setAssemblyGhosted(true, part);

  const select = $('fea-mode');
  select.innerHTML = '<option value="angle">at crank angle</option>'
    + '<option value="peak">cycle peak</option>'
    + '<option value="fatigue">fatigue SF</option>';
  select.value = 'angle';
  $('fea-legend').hidden = false;
  markRailButtons();

  const su = r.summary;
  $('rr-fea-result').innerHTML =
    `<b>${r.label}</b> &middot; ${su.elements.toLocaleString()} elements<br>`
    + `cycle peak ${(su.p99_5_von_mises_pa / 1e6).toFixed(0)} MPa `
    + `(99.5th pct) at ${su.peak_theta_deg.toFixed(0)}&deg;, allowable ${
      (su.allowable_pa / 1e6).toFixed(0)} MPa at ${su.temperature_c.toFixed(0)}&deg;C<br>`
    + `static SF <b>${su.static_safety_factor.toFixed(2)}</b> &middot; `
    + `fatigue SF <b>${su.min_fatigue_safety_factor.toFixed(2)}</b> (Goodman, `
    + `signed von Mises)<br>`
    + `equilibrium: force ${(su.equilibrium.max_relative_force * 100).toFixed(3)}%, `
    + `moment ${(su.equilibrium.max_relative_moment * 100).toFixed(2)}%`;
  await paintRailAngle();
}

window.repaintRailFea = function repaintRailFea() {
  if (!S.fea || !S.fea.rail) return;
  S.fea.mode = $('fea-mode').value;
  if (S.fea.mode === 'angle') { paintRailAngle(); return; }
  const f = S.fea.data.field;
  if (S.fea.mode === 'peak') {
    paintRail(f.peak_mpa, 0, f.range_mpa.p99_mpa,
      'cycle peak von Mises, every crank angle', 'MPa');
  } else {
    // fatigue: low is bad, so the ramp runs red at 0 to green at 3
    const inv = f.fatigue_sf.map((v) => 3 - Math.min(v, 3));
    paintRail(inv, 0, 3, 'Goodman fatigue safety factor', 'SF', true);
  }
};

async function paintRailAngle() {
  if (!S.fea || !S.fea.rail || S.fea.mode !== 'angle') return;
  const theta = S.sweep ? S.sweep.theta_deg[S.idx] : 0;
  let r;
  try {
    r = await api(`/api/railrod/fea/${S.fea.part}/at?theta_deg=${theta}`);
  } catch (e) { return; }
  if (!r.ok || !S.fea || S.fea.mode !== 'angle') return;
  const f = S.fea.data.field;
  const top = S.fea.data.allowable_mpa;
  const accel = (name) => /inertia|whip|axial/.test(name);
  const loads = r.loads.slice(0, 5).map((l) => `${l.case} ${
    accel(l.case) ? `${(l.value / 9.81).toFixed(0)} g` : kN(l.value)}`)
    .join(', ');
  paintRail(r.stress_mpa, 0, top,
    `von Mises at ${r.theta_deg.toFixed(0)}&deg;, scaled to yield; peak ${
      r.peak_mpa.toFixed(0)} MPa<br><span class="rr-muted">${loads}</span>`, 'MPa');
  void f;
}

function paintRail(values, lo, hi, caption, unit, inverted) {
  const colours = S.fea.mesh.geometry.getAttribute('color');
  const span = Math.max(hi - lo, 1e-9);
  for (let i = 0; i < values.length; i += 1) {
    const [r, g, b] = rampColour((values[i] - lo) / span);
    colours.array[i * 3] = r;
    colours.array[i * 3 + 1] = g;
    colours.array[i * 3 + 2] = b;
  }
  colours.needsUpdate = true;
  if (inverted) {
    $('legend-lo').textContent = '3+';
    $('legend-hi').textContent = '0';
  } else {
    $('legend-lo').textContent = `${lo.toFixed(0)}`;
    $('legend-hi').textContent = `${hi.toFixed(0)} ${unit}`;
  }
  $('fea-caption').innerHTML = `${S.fea.data.label}: ${caption}`;
}

function wireRailRod() {
  $('rr-enable').addEventListener('click', async () => {
    const on = RR.data && RR.data.enabled;
    await write({ 'railrod.enabled': !on },
      on ? 'back to the conventional rod' : 'switch to the rail rod');
    await loadRailRod();
  });
  $('rr-swing').addEventListener('input', (e) => {
    RR.swing = Number(e.target.value);
    $('rr-swing-value').textContent = `${RR.swing}°`;
    clearTimeout(RR.timer);
    RR.timer = setTimeout(loadSection, 60);
  });
  $('rr-forces').addEventListener('change', (e) => {
    RR.forces = e.target.value;
    loadSection();
  });
  $('rr-zoom').addEventListener('click', () => {
    RR.zoom = !RR.zoom;
    $('rr-zoom').textContent = RR.zoom ? 'big end' : 'notch';
    drawSection();
  });
  $('rr-couple').addEventListener('click', async () => {
    const b = $('rr-couple');
    b.disabled = true;
    note('solving the rails, receiver and clamp for the coupled network...');
    try {
      const r = await api('/api/railrod/coupled?elements=20000');
      if (!r.ok) toast(`<strong>Coupling failed:</strong> ${r.error}`, true);
    } finally {
      b.disabled = false;
      note('');
    }
    await loadRailRod();
  });
}

wireRailRod();
