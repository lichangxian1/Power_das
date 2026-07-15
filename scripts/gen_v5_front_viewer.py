#!/usr/bin/env python3
"""v5 前沿回放查看器：把 run 下所有前沿快照嵌进一个自包含 HTML，拖动进度条
按 ep 回放 area-vs-MRED / power-vs-MRED 两幅散点图。

快照来源（按 ep 去重合并）：
  <root>/<subrun>/logs/save_iterN/front.json        （save_freq 粒度，ep=N+1）
  <root>/<subrun>/logs/front_hist/front_epNNNN.json （front_dump_freq 粒度）
展示口径：
  * 主 run（GA）各 seg 档案点按子线着色，包络接成一条合并前沿（并集非支配，橙色）；
  * --compare label=path 可叠加对照 run（如 greedy A/B），各成一组独立前沿线；
  * 背景基线：纯截断 Dadda 种子（主 run 快照内 n_cells==0 每 k 首次入档）+
    纯截断 arith（--trunc_dir 的 k*/best_info.json，与 plot_v5_progress 同口径）；
  * 淡色参照 = 各组最终前沿，可开关。单文件零依赖，浏览器直接打开。

用法: python scripts/gen_v5_front_viewer.py <fronts_root> [out.html]
        [--compare greedy=outputs/2026-07-13_v5r2_greedy]
        [--trunc_dir outputs/2026-07-09_mred_trunc_baseline]
"""
import argparse
import json
import os
import re

# 参考调色板（已过 CVD/对比度校验）
DOT_COLORS = [("#2a78d6", "#3987e5"), ("#1baf7a", "#199e70")]   # 主组档案点：blue, aqua
MAIN_FRONT = ("#eb6834", "#d95926")                             # 主组前沿线：orange
CMP_COLORS = [("#eda100", "#c98500"), ("#e87ba4", "#d55181")]   # 对照组：yellow, magenta
VIOLET = ("#4a3aa7", "#9085e9")
GRAY = ("#8a8985", "#93938f")


def collect_runs(root):
    """[(name, {ep: [entry]})]；save_iterN → ep N+1，front_epNNNN → ep N。"""
    cands = []
    if os.path.isdir(os.path.join(root, "logs")):
        cands.append((os.path.basename(root.rstrip("/")), os.path.join(root, "logs")))
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "logs")
        if os.path.isdir(p):
            cands.append((d, p))
    runs = []
    for name, base in cands:
        snaps = {}
        hist = os.path.join(base, "front_hist")
        if os.path.isdir(hist):
            for d in os.listdir(hist):
                m = re.match(r"front_ep(\d+)\.json$", d)
                if m:
                    snaps[int(m.group(1))] = json.load(open(os.path.join(hist, d)))
        for d in os.listdir(base):
            m = re.match(r"save_iter(\d+)$", d)
            p = os.path.join(base, d, "front.json")
            if m and os.path.exists(p):
                snaps.setdefault(int(m.group(1)) + 1, json.load(open(p)))
        if snaps:
            runs.append((name, dict(sorted(snaps.items()))))
    return runs


def load_trunc_arith(base):
    """纯截断 arith 基线：k*/best_info.json → [[mred, area, power_W, k]]（升 mred）。"""
    pts = []
    if not base or not os.path.isdir(base):
        return pts
    for d in sorted(os.listdir(base)):
        m = re.match(r"k(\d+)", d)
        p = os.path.join(base, d, "best_info.json")
        if not (m and os.path.exists(p)):
            continue
        bi = json.load(open(p))
        mred = (bi.get("measured_error") or {}).get("mred")
        sr = (bi.get("simulated_result") or [{}])[0]
        if mred is not None and sr.get("area") is not None:
            pts.append([float(mred), float(sr["area"]), float(sr["power"]),
                        int(m.group(1))])
    return sorted(pts)


def dadda_staircase(runs):
    """纯截断 Dadda 种子阶梯：每 k 取**首次入档**的 n_cells==0 条目（同帧多条取
    最小面积）。全程取最小会混入训练中后期搜到的更优纯截断布线——那是 v5 的
    成果不是种子基线（用户 07-14 指正）。"""
    ks, first_ep = {}, {}
    for _name, snaps in runs:
        for ep in sorted(snaps):
            for e in snaps[ep]:
                if e.get("n_cells") != 0:
                    continue
                k = e.get("k", -1)
                if k in ks and first_ep[k] != ep:
                    continue          # 已有更早帧的记录，后续帧不再更新
                if k not in ks or e["area"] < ks[k][1]:
                    ks[k] = [e["mred"], e["area"], e["power"], k]
                    first_ep[k] = ep
    return sorted(ks.values())


def pack_run(name, group, cl, cd, snaps, static=False):
    if static:                      # 只保留最终快照，且回放全程恒显（静态参照）
        last = max(snaps)
        snaps = {last: snaps[last]}
    return {
        "name": name, "group": group, "cl": cl, "cd": cd, "static": static,
        "snaps": [{"ep": ep,
                   "pts": [[e["mred"], e["area"], e["power"],
                            e.get("k", -1), e.get("n_cells", -1), e.get("bin", -1)]
                           for e in ents]}
                  for ep, ents in snaps.items()],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--compare", action="append", default=[],
                    metavar="LABEL=PATH", help="叠加对照 run（可重复），如 greedy=outputs/xxx")
    ap.add_argument("--compare_final", action="append", default=[],
                    metavar="LABEL=PATH",
                    help="叠加对照 run 但只画其最终前沿（静态参照线，回放全程恒显）")
    ap.add_argument("--trunc_dir", default="outputs/2026-07-09_mred_trunc_baseline")
    ap.add_argument("--dadda_json", default="outputs/dadda_seed_staircase.json",
                    help="持久化种子阶梯 JSON（种子=教科书构造，跨战役通用；温启动 run"
                         "自己不评种子，一律从这里取。空串=退回从快照提取。新冷启动战役"
                         "覆盖新 k 时用 README 里的导出命令重生成，不自动写回——温启动"
                         "run 的零 cell 点是搜索成果，混入会污染种子口径）")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "front_viewer.html")

    main_runs = collect_runs(a.root)
    if not main_runs:
        raise SystemExit(f"未在 {a.root} 下找到任何 front.json / front_ep*.json")

    title = os.path.basename(a.root.rstrip("/"))
    main_label = title.split("_")[-2] if "_" in title else title   # 如 v6/v5r2
    data = {"title": title, "runs": [],
            "groups": [{"key": "GA",
                        "label": f"{main_label} 合并前沿 (seg 并集非支配)",
                        "cl": MAIN_FRONT[0], "cd": MAIN_FRONT[1]}],
            "baselines": []}
    for i, (name, snaps) in enumerate(main_runs):
        cl, cd = DOT_COLORS[i % len(DOT_COLORS)]
        data["runs"].append(pack_run(name, "GA", cl, cd, snaps))
        print(f"  GA/{name}: {len(snaps)} snapshots, ep {min(snaps)}..{max(snaps)}")

    all_for_dadda = list(main_runs)
    specs = ([(s, False) for s in a.compare]
             + [(s, True) for s in a.compare_final])
    for j, (spec, final_only) in enumerate(specs):
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--compare 需要 LABEL=PATH 形式，收到: {spec}")
        cl, cd = CMP_COLORS[j % len(CMP_COLORS)]
        data["groups"].append({"key": label,
                               "label": f"{label} 最终前沿" if final_only
                               else f"{label} 前沿",
                               "cl": cl, "cd": cd})
        cruns = collect_runs(path)
        if not cruns:
            print(f"  [warn] 对照 {label} 目录无快照，跳过: {path}")
            continue
        all_for_dadda.extend(cruns)
        for name, snaps in cruns:
            rname = label if len(cruns) == 1 else f"{label}:{name}"
            data["runs"].append(pack_run(rname, label, cl, cd, snaps,
                                         static=final_only))
            print(f"  {label}/{name}: {len(snaps)} snapshots, "
                  f"ep {min(snaps)}..{max(snaps)}"
                  + ("  [仅最终]" if final_only else ""))

    # 种子阶梯：持久化 JSON 为准（种子 = 教科书确定性构造，跨战役通用；温启动 run
    # 自己不评种子）。不从当前快照自动并入——温启动 run 的零 cell 点是搜索成果，
    # 不是种子，混入即污染口径。JSON 缺失才退回从快照提取（冷启动 run 可用）。
    if a.dadda_json and os.path.exists(a.dadda_json):
        dd = sorted(json.load(open(a.dadda_json))["pts"], key=lambda p: p[0])
    else:
        dd = dadda_staircase(all_for_dadda)
    if dd:
        data["baselines"].append({"name": "纯截断 Dadda 种子 (首次入档)", "kind": "dadda",
                                  "cl": GRAY[0], "cd": GRAY[1], "pts": dd})
        print(f"  baseline Dadda: {len(dd)} pts "
              f"(k{dd[0][3]:02d}..k{dd[-1][3]:02d})")
    ta = load_trunc_arith(a.trunc_dir)
    if ta:
        data["baselines"].append({"name": "纯截断 arith (mred C*)", "kind": "arith",
                                  "cl": VIOLET[0], "cd": VIOLET[1], "pts": ta})
        print(f"  baseline trunc-arith: {len(ta)} pts ({a.trunc_dir})")
    elif a.trunc_dir:
        print(f"  [warn] 纯截断 arith 基线目录不可用，跳过: {a.trunc_dir}")

    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(out, "w") as f:
        f.write(html)
    print("saved ->", out)


TEMPLATE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v5 front replay</title>
<style>
  :root {
    --surface: #fcfcfb; --panel: #ffffff; --text-1: #0b0b0b; --text-2: #52514e;
    --grid: #e8e7e3; --axis: #d5d4cf; --ring: #ffffff;
  }
  @media (prefers-color-scheme: dark) {
    :root { --surface:#1a1a19; --panel:#222221; --text-1:#ffffff; --text-2:#c3c2b7;
            --grid:#31312f; --axis:#414140; --ring:#222221; }
  }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--surface); color: var(--text-1);
         font: 14px/1.5 system-ui, "PingFang SC", "Microsoft YaHei", sans-serif;
         padding: 20px 24px 40px; }
  h1 { font-size: 18px; font-weight: 650; }
  .sub { color: var(--text-2); font-size: 12.5px; margin: 2px 0 14px; }
  .ctrl { display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
          padding: 10px 14px; background: var(--panel); border: 1px solid var(--grid);
          border-radius: 10px; margin-bottom: 16px; }
  .ctrl button { font: inherit; color: var(--text-1); background: none;
                 border: 1px solid var(--axis); border-radius: 8px;
                 padding: 4px 14px; cursor: pointer; }
  .ctrl button:hover { border-color: var(--text-2); }
  #slider { flex: 1; min-width: 220px; accent-color: #2a78d6; }
  #eplab { font-variant-numeric: tabular-nums; font-weight: 650; min-width: 130px; }
  label.tog { color: var(--text-2); font-size: 12.5px; display: flex;
              align-items: center; gap: 5px; cursor: pointer; }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 0 2px 10px;
            color: var(--text-2); font-size: 12.5px; }
  .legend .key { display: inline-block; width: 18px; height: 0;
                 border-top: 2.5px solid; vertical-align: middle; margin-right: 6px;
                 border-radius: 2px; }
  .legend .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
                 vertical-align: -1px; margin-right: 6px; }
  .legend .sq { display: inline-block; width: 8px; height: 8px; border-radius: 2px;
                vertical-align: -1px; margin-right: 6px; }
  .wrap { display: flex; gap: 16px; flex-wrap: wrap; }
  .card { flex: 1 1 480px; min-width: 380px; background: var(--panel);
          border: 1px solid var(--grid); border-radius: 12px; padding: 14px 14px 8px; }
  .card h2 { font-size: 13.5px; font-weight: 650; margin: 0 0 2px 6px; }
  .card .yl { color: var(--text-2); font-size: 11.5px; margin: 0 0 6px 6px; }
  svg { display: block; width: 100%; height: auto; }
  .tt { position: fixed; pointer-events: none; z-index: 9; display: none;
        background: var(--panel); border: 1px solid var(--axis); border-radius: 8px;
        padding: 7px 10px; font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.18);
        max-width: 270px; }
  .tt b { font-size: 13px; }
  .tt .k { display: inline-block; width: 14px; border-top: 2.5px solid;
           vertical-align: middle; margin-right: 5px; border-radius: 2px; }
  details { margin-top: 18px; }
  summary { cursor: pointer; color: var(--text-2); font-size: 13px; }
  table { border-collapse: collapse; margin-top: 10px; font-size: 12.5px;
          font-variant-numeric: tabular-nums; }
  th, td { padding: 3px 12px 3px 0; text-align: right; }
  th { color: var(--text-2); font-weight: 600; border-bottom: 1px solid var(--axis); }
  td:first-child, th:first-child { text-align: left; }
</style>
</head>
<body>
<h1 id="title"></h1>
<div class="sub">拖动进度条回放训练档案（ParetoArchive 快照）。每组一条前沿线
（组内各 seg 并集非支配），档案点按子线着色；背景 = 纯截断基线（灰=Dadda 种子
首次入档，紫=arith）。x 轴 = verilator 实测 MRED（log）；面积/功耗为环内 DC 口径，
XA 终审另算。</div>

<div class="ctrl">
  <button id="play">▶ 播放</button>
  <input id="slider" type="range" min="0" value="0" step="1">
  <span id="eplab"></span>
  <label class="tog"><input type="checkbox" id="ghost" checked>显示最终前沿（淡色参照）</label>
</div>
<div class="legend" id="legend"></div>
<div class="wrap">
  <div class="card"><h2>面积 vs MRED</h2><div class="yl">DC area (µm²)</div>
    <svg id="svg0" viewBox="0 0 640 430"></svg></div>
  <div class="card"><h2>功耗 vs MRED</h2><div class="yl">DC power (mW, 环内代理)</div>
    <svg id="svg1" viewBox="0 0 640 430"></svg></div>
</div>
<details id="tblbox"><summary>数据表（当前 ep 全部档案点）</summary>
  <div style="overflow-x:auto"><table id="tbl"></table></div></details>
<div class="tt" id="tt"></div>

<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const dark = matchMedia("(prefers-color-scheme: dark)");
const col = r => dark.matches ? r.cd : r.cl;

document.getElementById("title").textContent = "v5 前沿回放 — " + DATA.title;
document.title = "v5 front replay — " + DATA.title;

/* ---- ep 轴：非静态系列快照 ep 的并集；每系列取 <=当前ep 的最新快照；
        static 系列（仅最终参照）恒显最后一帧、不参与时间轴 ---- */
const epList = [...new Set(DATA.runs.filter(r => !r.static)
  .flatMap(r => r.snaps.map(s => s.ep)))].sort((a,b)=>a-b);
const snapAt = (r, ep) => {
  if (r.static) return r.snaps[r.snaps.length - 1];
  let s = null; for (const x of r.snaps) { if (x.ep <= ep) s = x; else break; } return s;
};
const groupRuns = g => DATA.runs.filter(r => r.group === g.key);
const mergedPts = (g, ep) => groupRuns(g).flatMap(r => { const s = snapAt(r, ep); return s ? s.pts : []; });

/* ---- 全局定轴（含基线点，回放期间坐标不动） ---- */
const all = DATA.runs.flatMap(r => r.snaps.flatMap(s => s.pts))
  .concat(DATA.baselines.flatMap(b => b.pts));
const xmin = Math.min(...all.map(p => p[0])), xmax = Math.max(...all.map(p => p[0]));
const PANELS = [
  { svg: document.getElementById("svg0"), yi: 1, sc: 1,    unit: "µm²" },
  { svg: document.getElementById("svg1"), yi: 2, sc: 1000, unit: "mW"  },
];
const M = { l: 62, r: 18, t: 10, b: 44 }, W = 640, H = 430;
const lx = Math.log10, X = v => M.l + (lx(v) - lx(xmin)) / (lx(xmax) - lx(xmin)) * (W - M.l - M.r);
for (const p of PANELS) {
  const ys = all.map(q => q[p.yi] * p.sc);
  const lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * 0.06 || 1;
  p.y0 = Math.max(0, lo - pad); p.y1 = hi + pad;
  p.Y = v => H - M.b - (v * p.sc - p.y0) / (p.y1 - p.y0) * (H - M.t - M.b);
}

/* ---- 非支配包络：按 mred 升序保 y 严格递减（与 plot_v5_progress 同口径） ---- */
function envelope(pts, yi) {
  const s = [...pts].sort((a,b) => a[0]-b[0]); const env = []; let best = 1/0;
  for (const p of s) if (p[yi] < best) { env.push(p); best = p[yi]; }
  return env;
}

function el(tag, at, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in at) e.setAttribute(k, at[k]);
  if (parent) parent.appendChild(e); return e;
}
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const SUP = {"-":"⁻","0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"};
const pow10 = e => "10" + String(e).split("").map(c => SUP[c]).join("");

function drawStatic(p) {
  p.svg.textContent = "";
  const g = el("g", {}, p.svg);
  for (let e = Math.ceil(lx(xmin)); e <= Math.floor(lx(xmax)); e++) {   // x 十进网格
    const x = X(10 ** e);
    el("line", { x1: x, y1: M.t, x2: x, y2: H - M.b, stroke: css("--grid"), "stroke-width": 1 }, g);
    el("text", { x, y: H - M.b + 18, "text-anchor": "middle", fill: css("--text-2"), "font-size": 11 }, g)
      .textContent = pow10(e);
  }
  const span = p.y1 - p.y0, step = niceStep(span / 5);
  for (let v = Math.ceil(p.y0 / step) * step; v <= p.y1 + 1e-9; v += step) {
    const y = H - M.b - (v - p.y0) / span * (H - M.t - M.b);
    el("line", { x1: M.l, y1: y, x2: W - M.r, y2: y, stroke: css("--grid"), "stroke-width": 1 }, g);
    el("text", { x: M.l - 8, y: y + 4, "text-anchor": "end", fill: css("--text-2"), "font-size": 11 }, g)
      .textContent = fmt(v);
  }
  el("line", { x1: M.l, y1: H - M.b, x2: W - M.r, y2: H - M.b, stroke: css("--axis"), "stroke-width": 1 }, g);
  el("line", { x1: M.l, y1: M.t, x2: M.l, y2: H - M.b, stroke: css("--axis"), "stroke-width": 1 }, g);
  el("text", { x: (M.l + W - M.r) / 2, y: H - 8, "text-anchor": "middle", fill: css("--text-2"), "font-size": 11.5 }, g)
    .textContent = "real MRED (log)";
  p.baseG = el("g", {}, p.svg);    // 基线背景层
  p.ghostG = el("g", {}, p.svg);   // 最终前沿参照
  p.dataG = el("g", {}, p.svg);    // 当前 ep 数据层
}
function niceStep(raw) {
  const m = 10 ** Math.floor(lx(raw)); const r = raw / m;
  return (r <= 1 ? 1 : r <= 2 ? 2 : r <= 5 ? 5 : 10) * m;
}
const fmt = v => Math.abs(v) >= 1000 ? Math.round(v).toLocaleString("en")
  : Math.abs(v) >= 10 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toPrecision(2);
const fx = v => v.toExponential(2);

let staticHits = [];   // 基线点命中区（一次性）
function drawBaselines() {
  staticHits = [];
  for (const p of PANELS) {
    p.baseG.textContent = "";
    for (const b of DATA.baselines) {
      const c = col(b);
      el("polyline", { points: b.pts.map(q => X(q[0]) + "," + p.Y(q[p.yi])).join(" "),
        fill: "none", stroke: c, "stroke-width": 1.8, opacity: 0.55,
        "stroke-linejoin": "round", "stroke-linecap": "round" }, p.baseG);
      for (const q of b.pts) {
        const x = X(q[0]), y = p.Y(q[p.yi]);
        if (b.kind === "dadda")
          el("circle", { cx: x, cy: y, r: 3, fill: css("--panel"),
            stroke: c, "stroke-width": 1.5, opacity: 0.7 }, p.baseG);
        else
          el("rect", { x: x - 2.8, y: y - 2.8, width: 5.6, height: 5.6, rx: 1.5,
            fill: c, opacity: 0.7 }, p.baseG);
        staticHits.push({ svg: p.svg, x, y, base: b, pt: q, panel: p });
      }
    }
  }
}

/* ---- 淡色参照：各组最终前沿 ---- */
function drawGhost() {
  const last = epList[epList.length - 1];
  for (const p of PANELS) {
    p.ghostG.textContent = "";
    if (!showGhost.checked) continue;
    for (const g of DATA.groups) {
      if (groupRuns(g).every(r => r.static)) continue;   // 静态组本体即最终前沿
      const env = envelope(mergedPts(g, last), p.yi);
      if (!env.length) continue;
      el("polyline", { points: env.map(q => X(q[0]) + "," + p.Y(q[p.yi])).join(" "),
        fill: "none", stroke: col(g), "stroke-width": 2.5, opacity: 0.25,
        "stroke-linejoin": "round", "stroke-linecap": "round" }, p.ghostG);
    }
  }
}

/* ---- 每帧重画数据层：每组一条包络线（压在点上）+ 档案点 ---- */
let hitPts = [];
function draw(idx) {
  const ep = epList[idx];
  document.getElementById("eplab").textContent = "ep " + ep + " / " + epList[epList.length - 1];
  hitPts = [];
  for (const p of PANELS) {
    p.dataG.textContent = "";
    for (const r of DATA.runs) {
      const s = snapAt(r, ep);
      if (!s) continue;
      for (const q of s.pts) {
        const x = X(q[0]), y = p.Y(q[p.yi]);
        el("circle", { cx: x, cy: y, r: r.static ? 3 : 3.5, fill: col(r),
          stroke: css("--ring"), "stroke-width": 1.5,
          opacity: r.static ? 0.4 : 1 }, p.dataG);       // 历史参照淡显
        hitPts.push({ svg: p.svg, x, y, run: r, pt: q, panel: p });
      }
    }
    for (const g of DATA.groups) {                 // 前沿线压在点上，不被圆点遮挡
      const env = envelope(mergedPts(g, ep), p.yi);
      if (!env.length) continue;
      const isStatic = groupRuns(g).every(r => r.static);
      el("polyline", { points: env.map(q => X(q[0]) + "," + p.Y(q[p.yi])).join(" "),
        fill: "none", stroke: col(g), "stroke-width": isStatic ? 2 : 2.5,
        opacity: isStatic ? 0.4 : 1,
        "stroke-linejoin": "round", "stroke-linecap": "round",
        "pointer-events": "none" }, p.dataG);
    }
  }
  const rows = [];
  for (const r of DATA.runs) {
    const s = snapAt(r, ep);
    if (s) for (const q of s.pts) rows.push([r.name, q[5], q[0], q[1], q[2], q[3], q[4]]);
  }
  rows.sort((a, b) => a[2] - b[2]);
  const tb = document.getElementById("tbl");
  tb.innerHTML =
    "<tr><th>series</th><th>bin</th><th>MRED</th><th>area µm²</th><th>power mW</th><th>k</th><th>cells</th></tr>";
  for (const w of rows) {
    const tr = document.createElement("tr");
    [w[0], w[1], fx(w[2]), fmt(w[3]), fmt(w[4] * 1000), w[5], w[6]].forEach(v => {
      const td = document.createElement("td"); td.textContent = v; tr.appendChild(td);
    });
    tb.appendChild(tr);
  }
}

/* ---- 图例 ---- */
function legend() {
  const lg = document.getElementById("legend"); lg.textContent = "";
  const item = (node, label) => { const s = document.createElement("span"); s.append(node, label); lg.appendChild(s); };
  const key = c => { const k = document.createElement("span"); k.className = "key"; k.style.borderTopColor = c; return k; };
  for (const g of DATA.groups) {
    const k = key(col(g));
    const gr = groupRuns(g);
    if (gr.length && gr.every(r => r.static)) k.style.opacity = 0.45;
    item(k, g.label);
  }
  for (const r of DATA.runs) {
    if (DATA.groups.some(g => g.key === r.group && g.key !== "GA")) continue;  // 对照组点色=线色，不重复列
    const d = document.createElement("span"); d.className = "dot"; d.style.background = col(r);
    item(d, r.name + " 档案点");
  }
  for (const b of DATA.baselines) {
    if (b.kind === "dadda") item(key(col(b)), b.name);
    else { const sq = document.createElement("span"); sq.className = "sq"; sq.style.background = col(b); item(sq, b.name); }
  }
  const gk = key(css("--text-2")); gk.style.opacity = 0.45;
  item(gk, "淡色 = 各组最终前沿参照");
}

/* ---- 悬停：最近点命中（半径 26px；档案点 + 基线点） ---- */
const tt = document.getElementById("tt");
let hl = null;
for (const p of PANELS) {
  p.svg.addEventListener("pointermove", ev => {
    const r = p.svg.getBoundingClientRect(), sx = W / r.width;
    const mx = (ev.clientX - r.left) * sx, my = (ev.clientY - r.top) * (H / r.height);
    let best = null, bd = 26 * sx;
    for (const h of hitPts.concat(staticHits)) {
      if (h.svg !== p.svg) continue;
      const d = Math.hypot(h.x - mx, h.y - my);
      if (d < bd) { bd = d; best = h; }
    }
    if (hl) { hl.remove(); hl = null; }
    if (!best) { tt.style.display = "none"; return; }
    const src = best.run || best.base;
    hl = el("circle", { cx: best.x, cy: best.y, r: 7.5, fill: "none",
      stroke: col(src), "stroke-width": 2 }, best.panel.dataG);
    const q = best.pt;
    tt.textContent = "";
    const head = document.createElement("div");
    const k = document.createElement("span"); k.className = "k"; k.style.borderTopColor = col(src);
    const bb = document.createElement("b");
    bb.textContent = (best.panel.yi === 1 ? fmt(q[1]) + " µm²" : fmt(q[2] * 1000) + " mW");
    head.append(k, bb); tt.appendChild(head);
    const l2 = document.createElement("div");
    l2.textContent = "MRED " + fx(q[0]) + " · k" + String(q[3]).padStart(2, "0")
      + (best.run ? " · cells " + q[4] + " · bin " + q[5] : "");
    l2.style.color = css("--text-2"); tt.appendChild(l2);
    const l3 = document.createElement("div");
    l3.textContent = best.run
      ? best.run.name + (q[4] === 0 ? " · 纯截断(0 cells)" : "") : best.base.name;
    l3.style.color = css("--text-2"); tt.appendChild(l3);
    tt.style.display = "block";
    tt.style.left = Math.min(ev.clientX + 14, innerWidth - 290) + "px";
    tt.style.top = (ev.clientY + 14) + "px";
  });
  p.svg.addEventListener("pointerleave", () => {
    tt.style.display = "none"; if (hl) { hl.remove(); hl = null; }
  });
}

/* ---- 控件 ---- */
const slider = document.getElementById("slider");
slider.max = epList.length - 1;
slider.addEventListener("input", () => { stop(); draw(+slider.value); });
const playBtn = document.getElementById("play");
let timer = null;
function stop() { if (timer) { clearInterval(timer); timer = null; playBtn.textContent = "▶ 播放"; } }
playBtn.addEventListener("click", () => {
  if (timer) return stop();
  playBtn.textContent = "⏸ 暂停";
  if (+slider.value >= epList.length - 1) slider.value = 0;
  timer = setInterval(() => {
    if (+slider.value >= epList.length - 1) return stop();
    slider.value = +slider.value + 1; draw(+slider.value);
  }, 500);
});
addEventListener("keydown", ev => {
  if (ev.key === "ArrowRight" && +slider.value < epList.length - 1) { stop(); slider.value = +slider.value + 1; draw(+slider.value); }
  if (ev.key === "ArrowLeft" && +slider.value > 0) { stop(); slider.value = +slider.value - 1; draw(+slider.value); }
  if (ev.key === " ") { ev.preventDefault(); playBtn.click(); }
});
const showGhost = document.getElementById("ghost");
showGhost.addEventListener("change", drawGhost);

function renderAll() {
  for (const p of PANELS) drawStatic(p);
  legend(); drawBaselines(); drawGhost(); draw(+slider.value);
}
dark.addEventListener("change", renderAll);
slider.value = epList.length - 1;   // 默认落在最新快照
renderAll();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
