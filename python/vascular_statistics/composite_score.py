"""
分割质量综合评分 —— 三元调和平均（重叠 + 拓扑 + 边界）

动机：Extraction 用 HM(Dice, clDice3D) 作 checkpoint_score，只含「重叠+拓扑」，
漏了边界精度（HD95）。血管分割喂下游骨架化，拓扑(clDice)最关键、其次重叠(Dice)、
再边界(HD95)。本模块把 HD95 经有界变换纳入，推广为三元调和平均：

    BoundaryScore = 1 / (1 + HD95/τ)          ∈ (0, 1]，HD95=0 → 1
    Score = HM(Dice, clDice3D, BoundaryScore) = 3 / (1/D + 1/C + 1/B)

调和平均特性：任一维度趋 0 → Score 趋 0（对短板敏感），优于算术/几何平均。
τ 为边界容差尺度（px），默认 2（与骨架匹配 δ 一致）；做 τ∈{1,2,4} 敏感性检查。

注：此评分可跨项目复用（含 Extraction 模型选型），见两项目 backlog。
"""

from typing import Optional


def boundary_score(hd95: float, tau: float = 2.0) -> float:
    """HD95(px, 越小越好, 无界) → 有界边界分 (0,1]。HD95=0→1, HD95→∞→0。"""
    if hd95 is None or hd95 < 0:
        return 0.0
    return 1.0 / (1.0 + hd95 / tau)


def composite_score(
    dice: float,
    cl_dice_3d: float,
    hd95: Optional[float] = None,
    tau: float = 2.0,
) -> float:
    """三元调和平均综合评分。

    参数：
        dice:       Dice 重叠度 ∈ [0,1]
        cl_dice_3d: 3D centerline Dice（拓扑）∈ [0,1]
        hd95:       HD95 边界距离（px）。None → 退化为二元 HM(Dice, clDice3D)
        tau:        边界分尺度（px），默认 2

    返回：综合评分 ∈ [0,1]。任一分量 ≤0 → 0。
    """
    if hd95 is None:
        vals = [dice, cl_dice_3d]
    else:
        vals = [dice, cl_dice_3d, boundary_score(hd95, tau)]
    if any(v is None or v <= 0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def _selftest() -> None:
    """算法正确性自检（3.2 要求）。"""
    import math
    # 完美：全 1, HD95=0 → 1
    assert abs(composite_score(1.0, 1.0, 0.0) - 1.0) < 1e-9
    # 二元退化 == HM(Dice, clDice3D)
    d, c = 0.8373, 0.8695
    hm2 = 2 * d * c / (d + c)
    assert abs(composite_score(d, c, None) - hm2) < 1e-9
    # 任一分量为 0 → 0
    assert composite_score(0.0, 0.9, 1.0) == 0.0
    assert composite_score(0.9, 0.9, None) > 0
    # 边界分单调：HD95 越大分越低
    assert boundary_score(0) > boundary_score(2) > boundary_score(10)
    assert abs(boundary_score(2, tau=2) - 0.5) < 1e-9  # HD95=τ → 0.5
    # 三元 ≤ 二元（加入 BoundaryScore<1 必拉低）
    s3 = composite_score(d, c, 8.0, tau=2)   # HD95=8 → B=0.2
    s2 = composite_score(d, c, None)
    assert s3 < s2
    # 调和平均对短板敏感：B 很低时 Score 接近被 B 拖住
    assert composite_score(0.9, 0.9, 100.0, tau=2) < 0.1
    # τ 敏感性：τ 越大边界惩罚越轻 → Score 越高
    assert composite_score(d, c, 8.0, tau=4) > composite_score(d, c, 8.0, tau=2)
    print("composite_score 自检全部通过")


if __name__ == "__main__":
    _selftest()
