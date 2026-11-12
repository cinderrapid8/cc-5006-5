let PROJECT = null;

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json();
  if (!r.ok) throw new Error(body.error || r.statusText);
  return body;
}

function fmt(x, d = 3) { return Number(x).toFixed(d); }

async function loadProject() {
  PROJECT = await api("/api/project");
  const s = PROJECT.source;
  $("srcinfo").textContent =
    `源视频 ${s.width}x${s.height} @ ${s.fps}fps, ${s.duration_s}s`;

  // anchors table
  const at = $("anchors");
  at.innerHTML = "<tr><th>锚点</th><th>视频时刻 s</th><th>数据时刻 s</th><th>残差 s</th></tr>" +
    PROJECT.anchors.map((a, i) =>
      `<tr><td>${a.label}</td><td>${fmt(a.video_time)}</td>` +
      `<td>${fmt(a.data_time)}</td><td>${fmt(PROJECT.alignment.residuals[i], 4)}</td></tr>`
    ).join("");
  const al = PROJECT.alignment;
  $("aligninfo").textContent =
    `拟合 offset=${fmt(al.offset, 4)}s rate=${fmt(al.rate, 6)}；` +
    `误差范围 [${fmt(al.offset_lo, 4)}, ${fmt(al.offset_hi, 4)}]s —— ` +
    `该范围仅覆盖全部人工锚点，不是精度保证。`;

  // data versions
  const dv = $("data_version");
  dv.innerHTML = Object.entries(PROJECT.data_versions).map(([v, info]) =>
    `<option value="${v}">${v} (${info.sha256}) — ${info.meta.note || ""}</option>`
  ).join("");

  // offset slider bounds: anchor range padded by 0.5s
  const lo = al.offset_lo - 0.5, hi = al.offset_hi + 0.5;
  const off = $("offset");
  off.min = lo; off.max = hi; off.value = al.offset;
  $("offset_val").textContent = fmt(al.offset, 3);
  off.oninput = () => $("offset_val").textContent = fmt(off.value, 3);
  $("rate").value = al.rate.toFixed(6);

  buildFieldCfg();
  refreshLists();
  loadQc();
}

function buildFieldCfg() {
  const ver = $("data_version").value || Object.keys(PROJECT.data_versions)[0];
  const fields = PROJECT.data_versions[ver].fields;
  const box = $("fieldcfg");
  box.innerHTML = "<h4>字段重采样（按字段配置）</h4>";
  for (const [name, f] of Object.entries(fields)) {
    const div = document.createElement("div");
    div.className = "fieldrow";
    if (f.kind === "continuous") {
      div.innerHTML =
        `<label><input type="checkbox" id="vis_${name}" checked> ${f.label} [${f.unit}]</label>` +
        `<select id="method_${name}"><option value="linear">线性插值</option>` +
        `<option value="hold">保持上次</option></select>` +
        `<label>最大缺口 s <input type="number" id="gap_${name}" step="0.1" ` +
        `value="${name === "temperature" ? 1.5 : 0.4}" style="width:4em"></label>` +
        `<label>显示单位 <input type="text" id="unit_${name}" value="${f.unit}" style="width:4.5em"></label>`;
    } else {
      div.innerHTML =
        `<label><input type="checkbox" id="vis_${name}" checked> ${f.label}（事件标记）</label>`;
    }
    box.appendChild(div);
  }
}

function collectParams() {
  const ver = $("data_version").value;
  const fields = [];
  for (const [name, f] of Object.entries(PROJECT.data_versions[ver].fields)) {
    const vis = $(`vis_${name}`)?.checked;
    if (f.kind === "continuous") {
      fields.push({
        name, visible: vis,
        method: $(`method_${name}`).value,
        max_gap: parseFloat($(`gap_${name}`).value),
        display_unit: $(`unit_${name}`).value || null,
        decimals: 1, color: f.color,
      });
    } else {
      fields.push({ name, visible: vis, color: f.color });
    }
  }
  return {
    data_version: ver,
    offset: parseFloat($("offset").value),
    rate: parseFloat($("rate").value),
    layout: $("layout").value,
    window_s: parseFloat($("window_s").value),
    fields,
  };
}

async function startRender() {
  const btn = $("render_btn");
  btn.disabled = true;
  try {
    const params = collectParams();
    const r = await api("/api/render", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    $("render_status").textContent = `渲染 ${r.render_id}: ${r.status}`;
    if (r.status === "done") { refreshLists(); return; }
    const poll = setInterval(async () => {
      const st = await api(`/api/render/${r.render_id}`);
      $("render_status").textContent = `渲染 ${r.render_id}: ${st.status}`;
      if (st.status !== "running") {
        clearInterval(poll);
        refreshLists();
      }
    }, 2000);
  } catch (e) {
    $("render_status").textContent = "错误: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function refreshLists() {
  const p = await api("/api/project");
  PROJECT.renders = p.renders; PROJECT.exports = p.exports;
  $("renders").innerHTML =
    "<tr><th>id</th><th>数据</th><th>offset</th><th>视频</th><th>清单</th><th></th></tr>" +
    p.renders.filter(r => r.done).map(r =>
      `<tr><td>${r.render_id}</td><td>${r.data_version}</td>` +
      `<td>${fmt(r.offset, 3)}</td>` +
      `<td><a href="/files/renders/${r.render_id}/overlay.mp4" target="_blank">mp4</a></td>` +
      `<td><a href="/files/renders/${r.render_id}/manifest.csv" target="_blank">csv</a></td>` +
      `<td><button onclick="viewRender('${r.render_id}')">播放</button>` +
      `<button onclick="exportRender('${r.render_id}')">导出</button></td></tr>`
    ).join("");
  $("exports").innerHTML =
    "<tr><th>id</th><th>渲染</th><th>时间</th><th>备注</th><th>视频</th><th>清单</th></tr>" +
    p.exports.map(e =>
      `<tr><td>${e.export_id}</td><td>${e.render_id}</td><td>${e.exported_at}</td>` +
      `<td>${e.note || ""}</td>` +
      `<td><a href="/files/exports/${e.export_id}/overlay.mp4" target="_blank">mp4</a></td>` +
      `<td><a href="/files/exports/${e.export_id}/manifest.csv" target="_blank">csv</a></td></tr>`
    ).join("");
}

function viewRender(rid) {
  $("player").src = `/files/renders/${rid}/overlay.mp4`;
  $("player").play();
}

async function exportRender(rid) {
  const note = prompt("导出备注（导出后不可变）:", "");
  if (note === null) return;
  await api("/api/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ render_id: rid, note }),
  });
  refreshLists();
}

async function loadQc() {
  const qc = await api("/api/qc");
  $("qc").innerHTML = (qc.checks || []).map(c =>
    `<li class="${c.pass ? "pass" : "fail"}">${c.pass ? "✓" : "✗"} ${c.check}: ${c.detail}</li>`
  ).join("");
}

$("render_btn").onclick = startRender;
$("data_version").onchange = buildFieldCfg;
loadProject().then(initScrub);

// ---------- scrub preview: sparse display curve vs exact readings ----------

let scrubSeq = 0;          // only the newest request may paint
let scrubInflight = false;
let scrubPending = null;
let lastPreview = null;

function initScrub() {
  const duration = PROJECT.source.duration_s;
  const bar = $("scrub");
  bar.max = duration;
  bar.addEventListener("input", () => scheduleScrub(parseFloat(bar.value)));
  const player = $("player");
  player.addEventListener("timeupdate", () => {
    bar.value = player.currentTime;
    scheduleScrub(player.currentTime);
  });
}

function scheduleScrub(videoTime) {
  scrubPending = videoTime;
  if (scrubInflight) return;  // coalesce drag events; latest wins
  requestAnimationFrame(() => {
    if (scrubPending !== null) {
      const t = scrubPending;
      scrubPending = null;
      fetchPreview(t);
    }
  });
}

async function fetchPreview(videoTime) {
  const seq = ++scrubSeq;
  scrubInflight = true;
  try {
    const body = await api("/api/preview", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_time_s: videoTime, params: collectParams() }),
    });
    if (seq !== scrubSeq) return;  // stale response: never painted
    lastPreview = body;
    paintPreview(body);
  } catch (e) {
    $("scrub_meta").textContent = "预览失败: " + e.message;
  } finally {
    scrubInflight = false;
    if (scrubPending !== null) scheduleScrub(scrubPending);
  }
}

function paintPreview(pv) {
  $("scrub_meta").textContent =
    `视频 ${fmt(pv.video_time_s)}s → 数据 ${fmt(pv.data_time_s)}s；` +
    `素材版本 ${pv.material_token.version} (${pv.material_token.sha256.slice(0, 8)})`;

  // exact readings: same path as the rendered overlay / manifest
  $("scrub_readings").innerHTML = pv.readings.map(r => {
    if (r.active !== undefined) {
      return `<span class="reading ${r.active ? "evt-on" : ""}">` +
        `${r.label}: ${r.active ? "事件激活" : "—"}</span>`;
    }
    const txt = r.status === "ok"
      ? `${Number(r.value).toFixed(r.decimals)} ${r.unit}`
      : (r.status === "missing" ? "缺测 —" : "rollback");
    return `<span class="reading st-${r.status}">${r.label}: ${txt}</span>`;
  }).join("");

  drawCurves(pv);
}

function drawCurves(pv) {
  const cv = $("scrub_canvas");
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const lo = pv.data_time_s - pv.window_s, hi = pv.data_time_s;
  const x = (t) => (t - lo) / (hi - lo) * W;
  const cont = pv.curves.filter(c => c.segments);
  const rowH = cont.length ? H / cont.length : H;

  cont.forEach((c, row) => {
    const top = row * rowH, bottom = top + rowH - 18;
    const okPts = c.segments.filter(s => s.kind === "ok")
      .flatMap(s => s.points.map(p => p[1]));
    const hasOk = okPts.length > 0;
    const vmin = hasOk ? Math.min(...okPts) : 0;
    const vmax = hasOk ? Math.max(...okPts) : 1;
    const y = (v) => bottom - (v - vmin) / ((vmax - vmin) || 1) * (bottom - top - 10) - 4;

    ctx.fillStyle = "#20242c";
    ctx.fillRect(0, top, W, rowH);
    ctx.strokeStyle = "#555";
    ctx.strokeRect(0.5, top + 0.5, W - 1, rowH - 1);
    ctx.fillStyle = "#bbb";
    ctx.font = "12px sans-serif";
    ctx.fillText(`${c.name} [${c.display_unit}]`, 6, top + 13);

    for (const s of c.segments) {
      if (s.kind !== "ok") {
        ctx.fillStyle = s.kind === "ambiguous"
          ? "rgba(255,200,40,0.25)" : "rgba(255,60,60,0.22)";
        ctx.fillRect(x(s.t0), top + 18, Math.max(1, x(s.t1) - x(s.t0)), bottom - top - 18);
        continue;
      }
      ctx.strokeStyle = "#ffb000";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      s.points.forEach(([t, v], i) => {
        const px = x(t), py = y(v);
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      });
      ctx.stroke();
    }
  });

  // event ticks across the bottom
  const evts = pv.curves.filter(c => c.events).flatMap(c => c.events);
  ctx.strokeStyle = "#ff5555";
  for (const te of evts) {
    const px = x(te);
    ctx.beginPath();
    ctx.moveTo(px, H - 14);
    ctx.lineTo(px, H - 2);
    ctx.stroke();
  }
}
