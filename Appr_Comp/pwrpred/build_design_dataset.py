#!/usr/bin/env python
"""设计级数据集：全部 reeval_xa*.csv / ppa_xa.csv 的 XA 标签 + MUL.v 结构特征。"""
import glob, json, os, re, csv
import pandas as pd

OUT = "/tmp/claude-1000/-home-lee-Power-das/c973f85e-6eb5-4da8-b295-c00dfc39f184/scratchpad/pwrpred/design_xa.csv"

# cell 库：name -> (dyn, area)
def load_libs():
    m = {}
    lib = json.load(open("/home/lee/Power_das/Appr_Comp/library.json"))
    for name, c in lib["cells"].items():
        m[name] = (c["dyn_w"] * 1e3, c["area"])
    lib42 = json.load(open("/home/lee/Power_das/Appr_Comp/library42_native.json"))
    cells42 = lib42["cells"]
    it = cells42.items() if isinstance(cells42, dict) else ((c["name"], c) for c in cells42)
    for name, c in it:
        m[name] = (c["dyn_mw"], c["area"])
    anch = lib42["meta"]["anchors"]
    for k in ("FA", "HA", "CT42_BAL", "CT42_TPL", "CT42_FLAT"):
        if k in anch:
            m[k] = (anch[k]["dyn_mw"], anch[k]["area"])
    return m

INST_RE = re.compile(r"^\s*(FA|HA|CT42\w*|comp\d+n?_[0-9a-f]+)\s+\S+\s*\(", re.M)

def rtl_features(path, libmap):
    src = open(path).read()
    inst = INST_RE.findall(src)
    n_fa = sum(1 for i in inst if i == "FA")
    n_ha = sum(1 for i in inst if i == "HA")
    n_ct42e = sum(1 for i in inst if i.startswith("CT42"))
    n_c32 = sum(1 for i in inst if i.startswith("comp32"))
    n_c22 = sum(1 for i in inst if i.startswith("comp22"))
    n_c42 = sum(1 for i in inst if i.startswith("comp42"))
    sum_dyn = sum_area = 0.0; n_unk = 0
    for i in inst:
        if i in libmap:
            d, a = libmap[i]; sum_dyn += d; sum_area += a
        else:
            n_unk += 1
    n_pp = len(re.findall(r"^\s*assign pp_", src, re.M))
    n_pp_active = len(re.findall(r"^\s*assign pp_.*&", src, re.M))
    n_const0 = src.count("1'b0")
    n_const1 = src.count("1'b1")
    n_wire = len(re.findall(r"^\s*wire ", src, re.M))
    n_assign = len(re.findall(r"^\s*assign ", src, re.M))
    booth = 1 if ("booth" in src.lower() or re.search(r"~\s*\(?\s*a\[", src)) else 0
    return dict(n_fa=n_fa, n_ha=n_ha, n_ct42e=n_ct42e, n_c32=n_c32, n_c22=n_c22,
                n_c42=n_c42, n_cells=len(inst), n_unknown_cells=n_unk,
                sum_dyn_lib=sum_dyn, sum_area_lib=sum_area,
                n_pp=n_pp, n_pp_active=n_pp_active, n_const0=n_const0,
                n_const1=n_const1, n_wire=n_wire, n_assign=n_assign, booth=booth)

def main():
    libmap = load_libs()
    rows, seen = [], set()
    files = sorted(glob.glob("/home/lee/Power_das/outputs/*/reeval_xa*.csv")) + \
            sorted(glob.glob("/home/lee/Power_das/outputs/*/ppa_xa.csv"))
    for f in files:
        run = os.path.basename(os.path.dirname(f))
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "power_xa_mw" not in df.columns:
            continue
        for _, r in df.iterrows():
            des = str(r["design"])
            key = (run, des)
            if key in seen: continue
            if "success" in df.columns and str(r.get("success")) not in ("True", "1", "1.0", "nan"):
                if str(r.get("success")) != "nan": continue
            mv = os.path.join("/home/lee/Power_das/outputs", run, des, "MUL.v")
            if not os.path.exists(mv): continue
            p = r["power_xa_mw"]
            if not pd.notna(p) or p <= 0: continue
            seen.add(key)
            feat = rtl_features(mv, libmap)
            feat.update(run=run, design=des, power_xa_mw=float(p),
                        area_dc=float(r["area_dc"]) if "area_dc" in df.columns and pd.notna(r.get("area_dc")) else float("nan"),
                        delay=abs(float(r["delay"])) if "delay" in df.columns and pd.notna(r.get("delay")) else float("nan"))
            rows.append(feat)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} designs from {len(set(r['run'] for r in rows))} runs -> {OUT}")
    d = pd.DataFrame(rows)
    print(d.groupby("run").size().to_string())
    print("unknown cells total:", d.n_unknown_cells.sum())

if __name__ == "__main__":
    main()
