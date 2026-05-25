#ifndef VASCULAR_STATISTICS_DATA_IO_H
#define VASCULAR_STATISTICS_DATA_IO_H

#include <string>

namespace vessel_stats {

/// 从 Pajek 衍生的边表 + 节点表中统计血管形态学指标。
/// 输入文件格式：
///   edges_file.txt    — 以空白分隔的整数对（节点编号）
///   vertices_file.txt — 每节点 6 个值：[idx, type, x, y, z, radius]
/// 输出文件（生成于当前工作目录）：
///   generate_vessel.txt             — 各段节点序列
///   generate_vessel_radius.txt      — 各段平均半径
///   generate_vessel_path_length.txt — 各段路径长度
///   generate_vessel_tortuosity.txt  — 各段弯曲度
///   statistics_summary.txt          — 汇总统计
bool GenerateStatistics(const std::string& edges_file,
                        const std::string& vertices_file,
                        double volume);

}  // namespace vessel_stats

#endif  // VASCULAR_STATISTICS_DATA_IO_H
