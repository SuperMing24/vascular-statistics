"""
3D 骨架健康度 + 匹配度 —— LFD 输出 .pajek vs 金标准 .pajek（物理 μm 空间）

适配自 Extraction `skeleton_eval.py`（Chap_4 §4.7.2，原为 2D 逐切片掩码骨架），
本工具直接作用于 3D 图(.pajek)：

  健康度（仅预测骨架，无需 GT）：
    isolated        孤立节点（度=0）
    spurious_loops  伪环路 β₁ = E − N + C（第一 Betti 数）
    excess_branch   过量分支节点（度≥4）
    total_defects   三者之和

  匹配度（vs 金标准，δ 容差）：
    completeness = |{g∈G: ∃p∈P, d(g,p)≤δ}| / |G|   （recall）
    correctness  = |{p∈P: ∃g∈G, d(p,g)≤δ}| / |P|   （precision）
    quality      = |P+| / (|P| + |G| − |P+|)        （IoU 式）

⚠️ pixel/μm 影响（用户 2.1）：δ 是距离阈值。不同网格(LFD 输出 grid X vs 金标准
grid 363/382)必须在**物理 μm 空间**比较——节点坐标 × 有效间距后再算距离，δ 用 μm。
同网格时 spacing 可设 1 退化为 px。

用法：
  python scripts/skeleton_match_3d.py \\
    --pred out.pajek --gold gold.pajek \\
    --pred-spacing 1.932 1.932 2.0 --gold-spacing 1.932 1.932 2.0 \\
    --delta-um 2.0
"""

import argparse
import json
import re
from typing import Dict, List, Tuple

import numpy as np


# ── Pajek 解析 ────────────────────────────────────────────
_POS_RE = re.compile(r'pos\s+"\[([^\]]+)\]"')


def parse_pajek(path: str) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict[int, int]]:
    """解析 .pajek → (coords[N,3], edges[(pid_a,pid_b)], pid→index)。"""
    coords: List[List[float]] = []
    edges: List[Tuple[int, int]] = []
    pid2idx: Dict[int, int] = {}
    mode = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            low = s.lower()
            if low.startswith("*vertices"):
                mode = "v"
                continue
            if low.startswith("*edges") or low.startswith("*arcs"):
                mode = "e"
                continue
            if low.startswith("*"):
                mode = None
                continue
            if mode == "v":
                m = _POS_RE.search(s)
                if m:
                    xyz = [float(t) for t in m.group(1).split()][:3]
                    if len(xyz) == 3:
                        pid = int(s.split()[0])
                        pid2idx[pid] = len(coords)
                        coords.append(xyz)
            elif mode == "e":
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        edges.append((int(parts[0]), int(parts[1])))
                    except ValueError:
                        pass
    return np.asarray(coords, dtype=np.float64), edges, pid2idx


# ── 健康度（3D 图）────────────────────────────────────────
def skeleton_health(coords: np.ndarray, edges, pid2idx) -> Dict[str, int]:
    import networkx as nx

    n = len(coords)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for a, b in edges:
        if a in pid2idx and b in pid2idx:
            ia, ib = pid2idx[a], pid2idx[b]
            if ia != ib:
                g.add_edge(ia, ib)
    deg = dict(g.degree())
    isolated = sum(1 for i in range(n) if deg.get(i, 0) == 0)
    excess_branch = sum(1 for i in range(n) if deg.get(i, 0) >= 4)
    e = g.number_of_edges()
    c = nx.number_connected_components(g)
    spurious_loops = max(0, e - n + c)  # β₁
    return {
        "isolated": isolated,
        "spurious_loops": int(spurious_loops),
        "excess_branch": excess_branch,
        "total_defects": isolated + int(spurious_loops) + excess_branch,
        "n_nodes": n,
        "n_edges": e,
        "n_components": c,
    }


# ── 匹配度（3D 点云，物理 μm）─────────────────────────────
def skeleton_matching(
    pred: np.ndarray,
    gold: np.ndarray,
    pred_spacing=(1.0, 1.0, 1.0),
    gold_spacing=(1.0, 1.0, 1.0),
    delta_um: float = 2.0,
) -> Dict[str, float]:
    from scipy.spatial import cKDTree

    if len(pred) == 0 and len(gold) == 0:
        return {"completeness": 1.0, "correctness": 1.0, "quality": 1.0,
                "n_pred": 0, "n_gold": 0}
    if len(pred) == 0 or len(gold) == 0:
        return {"completeness": 0.0, "correctness": 0.0, "quality": 0.0,
                "n_pred": len(pred), "n_gold": len(gold)}

    p = pred * np.asarray(pred_spacing, dtype=np.float64)   # → μm
    gm = gold * np.asarray(gold_spacing, dtype=np.float64)

    d_p2g, _ = cKDTree(gm).query(p)
    d_g2p, _ = cKDTree(p).query(gm)
    matched_pred = int((d_p2g <= delta_um).sum())   # |P+|
    matched_gold = int((d_g2p <= delta_um).sum())

    completeness = matched_gold / len(gm)
    correctness = matched_pred / len(p)
    denom = len(p) + len(gm) - matched_pred
    quality = matched_pred / denom if denom > 0 else 0.0
    return {
        "completeness": float(completeness),
        "correctness": float(correctness),
        "quality": float(quality),
        "n_pred": len(p),
        "n_gold": len(gm),
    }


def evaluate(pred_path, gold_path, pred_spacing, gold_spacing, delta_um) -> Dict:
    pc, pe, pmap = parse_pajek(pred_path)
    gc, ge, gmap = parse_pajek(gold_path)
    return {
        "health_pred": skeleton_health(pc, pe, pmap),
        "matching": skeleton_matching(pc, gc, pred_spacing, gold_spacing, delta_um),
        "delta_um": delta_um,
        "pred_spacing": list(pred_spacing),
        "gold_spacing": list(gold_spacing),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred-spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    ap.add_argument("--gold-spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    ap.add_argument("--delta-um", type=float, default=2.0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    res = evaluate(args.pred, args.gold, args.pred_spacing, args.gold_spacing, args.delta_um)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
