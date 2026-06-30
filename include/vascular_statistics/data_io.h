#ifndef VASCULAR_STATISTICS_DATA_IO_H
#define VASCULAR_STATISTICS_DATA_IO_H

#include <string>

namespace vessel_stats {

/// 从 Pajek 衍生的边表 + 节点表中统计血管形态学指标。
/// 输入文件格式：
///   edges_file.txt    — 以空白分隔的整数对（节点编号）
///   vertices_file.txt — 每节点 6 个值：[idx, type, x, y, z, radius]
/// 输出文件（生成于当前工作目录）——全量、无过滤，过滤/分档交由 Python 子范围命令：
///   generate_vessel.txt             — 各段节点序列
///   generate_vessel_radius.txt      — 各段平均半径
///   generate_vessel_path_length.txt — 各段路径长度
///   generate_vessel_tortuosity.txt  — 各段弯曲度
///   statistics_summary.txt          — 全量汇总统计（无直径/节点数/P99 过滤）
/// spacing（sx,sy,sz, μm/体素）：三者均 > 0 时启用各向异性物理单位
/// （距离逐轴加权得 μm；半径→μm 标量 = (sx+sy)/2 XY 均值）。
/// 任一 <= 0（默认）→ legacy 各向同性模式（×2 长度 / ×4 直径），
/// 保持与历史输出口径一致。
/// radius_physical（方案B）：节点 r 已是物理 μm（骨架化用各向异性 EDT 算）→
/// r_scale=1.0，不再缩放半径（直径=2r）；位置仍用 spacing 算长度。
bool GenerateStatistics(const std::string& edges_file,
                        const std::string& vertices_file,
                        double volume,
                        double sx = -1.0,
                        double sy = -1.0,
                        double sz = -1.0,
                        bool radius_physical = false);

}  // namespace vessel_stats

#endif  // VASCULAR_STATISTICS_DATA_IO_H
