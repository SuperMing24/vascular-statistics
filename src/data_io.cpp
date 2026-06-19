#include "vascular_statistics/data_io.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

// 中位数（接收副本，内部排序）。
double Median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    if (n % 2 == 0) return (values[n / 2 - 1] + values[n / 2]) / 2.0;
    return values[n / 2];
}

// 算术平均。
double Mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double v : values) sum += v;
    return sum / static_cast<double>(values.size());
}

// MAD（中位数绝对偏差）离群计数：|0.6745 * (x - median) / MAD| > k。
// 单位无关；MAD 退化为 0（半数以上同值）时不标记任何离群。
int CountMadOutliers(const std::vector<double>& values, double k) {
    if (values.size() < 2) return 0;
    const double med = Median(values);
    std::vector<double> abs_dev;
    abs_dev.reserve(values.size());
    for (double v : values) abs_dev.push_back(std::fabs(v - med));
    const double mad = Median(abs_dev);
    if (mad < 1e-10) return 0;
    int count = 0;
    for (double v : values) {
        if (std::fabs(0.6745 * (v - med) / mad) > k) ++count;
    }
    return count;
}

}  // namespace

namespace vessel_stats {

bool GenerateStatistics(const std::string& edges_file,
                        const std::string& vertices_file,
                        double volume) {
    // ---------- 第 1 遍：打开边文件，找最大节点编号 ----------
    std::ifstream in(edges_file + ".txt");
    if (!in) {
        std::cerr << "打开边文件失败: " << edges_file << ".txt\n";
        return false;
    }

    int idx_max = 0;
    int val;
    while (in >> val) {
        if (val > idx_max) idx_max = val;
    }
    in.close();

    // ---------- 读取节点文件 ----------
    in.open(vertices_file + ".txt");
    if (!in) {
        std::cerr << "打开节点文件失败: " << vertices_file << ".txt\n";
        return false;
    }

    // 先一次性读入全部浮点数，再据实际行数定尺寸。
    // Q6 修复：vertices_info 尺寸取「edges 最大编号 idx_max」与「vertices
    // 实际行数」的较大值，防止存在孤立节点（度=0、不出现在任何边里）时
    // 按行号写入越界（原实现用 idx_max 定尺寸，孤立高编号节点会溢出）。
    std::vector<double> raw_vtx;
    double f;
    while (in >> f) raw_vtx.push_back(f);
    in.close();

    const int vtx_rows = static_cast<int>(raw_vtx.size()) / 6;
    const int n_nodes = (vtx_rows > idx_max) ? vtx_rows : idx_max;

    // VerticesInfo[i] = {x, y, z, radius}
    std::vector<std::vector<double>> vertices_info(n_nodes,
                                                   std::vector<double>(4, 0.0));

    for (size_t i = 0; i < raw_vtx.size(); ++i) {
        const int idx = static_cast<int>(i) / 6;
        const int col = static_cast<int>(i) % 6;
        if (col == 2)       // x
            vertices_info[idx][0] = raw_vtx[i];
        else if (col == 3)  // y
            vertices_info[idx][1] = raw_vtx[i];
        else if (col == 4)  // z
            vertices_info[idx][2] = raw_vtx[i];
        else if (col == 5)  // radius
            vertices_info[idx][3] = raw_vtx[i];
    }

    // ---------- 统计节点度数 ----------
    std::vector<int> degrees(idx_max, 0);
    in.open(edges_file + ".txt");
    if (!in) {
        std::cerr << "打开边文件失败 (第 2 遍): " << edges_file << ".txt\n";
        return false;
    }
    while (in >> val) {
        degrees[val - 1]++;  // 节点编号 1..N → 数组索引 0..N-1
    }
    in.close();

    // 根据度数分类节点：0=不存在, 1=端点, 2=内部节点, 3=分叉点
    std::vector<int> type(idx_max, 0);
    int degrees_sum = 0;
    for (int i = 0; i < idx_max; i++) {
        degrees_sum += degrees[i];
        if (degrees[i] >= 3)
            type[i] = 3;
        else
            type[i] = degrees[i];
    }

    // ---------- 读取边对 ----------
    in.open(edges_file + ".txt");
    if (!in) {
        std::cerr << "打开边文件失败 (第 3 遍): " << edges_file << ".txt\n";
        return false;
    }
    int total_count = 0;
    while (in >> val) total_count++;
    int pair_count = total_count / 2;
    std::cout << pair_count << " 对边\n";
    in.close();

    // edge_array[p] = {node1, node2, flag}
    // flag: 1 = 未遍历, 0 = 已遍历
    std::vector<std::vector<int>> edge_array(pair_count,
                                             std::vector<int>(3, 0));
    for (int i = 0; i < pair_count; i++) edge_array[i][2] = 1;

    in.open(edges_file + ".txt");
    if (!in) {
        std::cerr << "打开边文件失败 (第 4 遍): " << edges_file << ".txt\n";
        return false;
    }
    int record = 0;
    while (in >> val) {
        edge_array[record / 2][record % 2] = val;
        record++;
    }
    in.close();

    // ---------- 打开输出文件 ----------
    std::ofstream out1("generate_vessel.txt");
    std::ofstream out2("generate_vessel_radius.txt");
    std::ofstream out3("generate_vessel_path_length.txt");
    std::ofstream out4("generate_vessel_tortuosity.txt");
    std::ofstream out5("statistics_summary.txt");

    if (!out1 || !out2 || !out3 || !out4 || !out5) {
        std::cerr << "创建输出文件失败\n";
        return false;
    }

    out5 << "Vascular_Statistics — 单次运行统计\n";

    // ---------- 段遍历 ----------
    int select_node = 0;
    int previous_node, next_node;
    bool end_of_vessel;
    double avg_temp;
    // 有效段（直径>=10μm 且节点数>=3）的逐段值缓存，
    // 用于遍历结束后计算均值 + 中位数 + 离群标记（轨道 A）。
    std::vector<double> eff_radii;
    std::vector<double> eff_lengths;
    std::vector<double> eff_torts;

    double vessel_start_x, vessel_start_y, vessel_start_z;
    double vessel_end_x, vessel_end_y, vessel_end_z;
    double path_length;
    double path_direct_distance;
    int k;  // 段内节点计数

    while (degrees_sum != 0) {
        // 寻找下一个起点：端点(1)或分叉点(3)且仍有未遍历边
        select_node = 0;
        for (int i = 0; i < idx_max; i++) {
            if ((type[i] == 1 || type[i] == 3) && degrees[i] >= 1) {
                select_node = i + 1;
                break;
            }
        }
        // 保护：无合法起点时退出（图数据不一致，如孤立边或度数计数错误）
        if (select_node == 0) {
            std::cerr << "Warning: 无可遍历起点（剩余 degree_sum="
                      << degrees_sum << "），强制退出段遍历。\n";
            break;
        }

        out1 << select_node << " ";

        previous_node = select_node;
        vessel_start_x = vertices_info[select_node - 1][0];
        vessel_start_y = vertices_info[select_node - 1][1];
        vessel_start_z = vertices_info[select_node - 1][2];

        end_of_vessel = false;
        path_length = 0.0;
        avg_temp = vertices_info[previous_node - 1][3];
        k = 1;

        while (!end_of_vessel) {
            bool edge_found = false;
            for (int i = 0; i < pair_count; i++) {
                // 正向匹配
                if (edge_array[i][0] == previous_node && edge_array[i][2] != 0) {
                    edge_found = true;
                    next_node = edge_array[i][1];
                    avg_temp -= (avg_temp - vertices_info[next_node - 1][3]) / (k + 1);
                    k++;
                    degrees[previous_node - 1]--;
                    out1 << next_node << " ";
                    degrees[next_node - 1]--;
                    if (type[next_node - 1] == 1 || type[next_node - 1] == 3)
                        end_of_vessel = true;

                    double dx = vertices_info[previous_node - 1][0] - vertices_info[next_node - 1][0];
                    double dy = vertices_info[previous_node - 1][1] - vertices_info[next_node - 1][1];
                    double dz = vertices_info[previous_node - 1][2] - vertices_info[next_node - 1][2];
                    path_length += std::sqrt(dx * dx + dy * dy + dz * dz);

                    previous_node = next_node;
                    edge_array[i][2] = 0;
                    degrees_sum -= 2;

                    if (end_of_vessel) {
                        out1 << "\n";
                        out2 << avg_temp << "\n";
                        vessel_end_x = vertices_info[previous_node - 1][0];
                        vessel_end_y = vertices_info[previous_node - 1][1];
                        vessel_end_z = vertices_info[previous_node - 1][2];
                        dx = vessel_start_x - vessel_end_x;
                        dy = vessel_start_y - vessel_end_y;
                        dz = vessel_start_z - vessel_end_z;
                        path_direct_distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                        out3 << path_length << "\n";
                        // 防止直连距离为 0 时产生 inf/NaN（孤立的单节点段）
                        if (path_direct_distance < 1e-10) {
                            out4 << "1.0\n";
                        } else {
                            out4 << path_length / path_direct_distance << "\n";
                        }

                        if (k > 2 && avg_temp >= 2.5) {
                            eff_radii.push_back(avg_temp);
                            eff_lengths.push_back(path_length);
                            eff_torts.push_back(
                                path_direct_distance < 1e-10
                                    ? 1.0
                                    : path_length / path_direct_distance);
                        }
                        break;
                    }
                    break;  // 找到匹配边后退出 for 循环，while 循环以新 previous_node 继续
                }
                // 反向匹配
                if (edge_array[i][1] == previous_node && edge_array[i][2] != 0) {
                    edge_found = true;
                    next_node = edge_array[i][0];
                    avg_temp -= (avg_temp - vertices_info[next_node - 1][3]) / (k + 1);
                    k++;
                    degrees[previous_node - 1]--;
                    out1 << next_node << " ";
                    degrees[next_node - 1]--;
                    if (type[next_node - 1] == 1 || type[next_node - 1] == 3)
                        end_of_vessel = true;

                    double dx = vertices_info[previous_node - 1][0] - vertices_info[next_node - 1][0];
                    double dy = vertices_info[previous_node - 1][1] - vertices_info[next_node - 1][1];
                    double dz = vertices_info[previous_node - 1][2] - vertices_info[next_node - 1][2];
                    path_length += std::sqrt(dx * dx + dy * dy + dz * dz);

                    previous_node = next_node;
                    edge_array[i][2] = 0;
                    degrees_sum -= 2;

                    if (end_of_vessel) {
                        out1 << "\n";
                        out2 << avg_temp << "\n";
                        vessel_end_x = vertices_info[previous_node - 1][0];
                        vessel_end_y = vertices_info[previous_node - 1][1];
                        vessel_end_z = vertices_info[previous_node - 1][2];
                        dx = vessel_start_x - vessel_end_x;
                        dy = vessel_start_y - vessel_end_y;
                        dz = vessel_start_z - vessel_end_z;
                        path_direct_distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                        out3 << path_length << "\n";
                        if (path_direct_distance < 1e-10) {
                            out4 << "1.0\n";
                        } else {
                            out4 << path_length / path_direct_distance << "\n";
                        }

                        if (k > 2 && avg_temp >= 2.5) {
                            eff_radii.push_back(avg_temp);
                            eff_lengths.push_back(path_length);
                            eff_torts.push_back(
                                path_direct_distance < 1e-10
                                    ? 1.0
                                    : path_length / path_direct_distance);
                        }
                        break;
                    }
                    break;  // 找到匹配边后退出 for 循环，while 循环以新 previous_node 继续
                }
            }
            // 保护：当前节点无未遍历边（孤立节点或度数计数不一致）
            if (!edge_found) {
                if (degrees[previous_node - 1] > 0) {
                    degrees_sum -= degrees[previous_node - 1];
                    degrees[previous_node - 1] = 0;
                }
                end_of_vessel = true;
                out1 << "\n";
            }
        }
    }

    // ---------- 计算汇总指标（均值 + 中位数 + 离群标记，轨道 A）----------
    const int n_eff = static_cast<int>(eff_lengths.size());

    const double mean_radius = Mean(eff_radii);
    const double mean_length = Mean(eff_lengths);
    const double mean_tort = Mean(eff_torts);
    const double median_radius = Median(eff_radii);
    const double median_length = Median(eff_lengths);
    const double median_tort = Median(eff_torts);

    // 离群段标记（只标记不剔除）：
    //   MAD（k=3.5，相对/分布判据，单位无关）
    //   绝对兜底（600 体素 ≈ 1200μm 当前 ×2 口径，待轨道 B 校准为真 μm）
    const double kMadK = 3.5;
    const double kAbsLengthThresholdVoxel = 600.0;
    const int mad_outliers = CountMadOutliers(eff_lengths, kMadK);
    int abs_outliers = 0;
    for (double len : eff_lengths) {
        if (len > kAbsLengthThresholdVoxel) ++abs_outliers;
    }
    const double mad_ratio = n_eff > 0 ? 100.0 * mad_outliers / n_eff : 0.0;
    const double abs_ratio = n_eff > 0 ? 100.0 * abs_outliers / n_eff : 0.0;

    // ---------- 写汇总文件 ----------
    out5 << "平均直径 (μm): " << mean_radius * 4 << "\n"
         << "中位直径 (μm): " << median_radius * 4 << "\n"
         << "平均长度 (μm): " << mean_length * 2 << "\n"
         << "中位长度 (μm): " << median_length * 2 << "\n"
         << "段密度 (seg/mm³): " << static_cast<double>(n_eff) / volume << "\n"
         << "平均弯曲度: " << mean_tort << "\n"
         << "中位弯曲度: " << median_tort << "\n"
         << "有效段数: " << n_eff << "\n"
         << "MAD 离群段 (k=3.5): " << mad_outliers << " (" << mad_ratio << "%)\n"
         << "超绝对阈值段 (>600 体素, 待轨道B校准): " << abs_outliers
         << " (" << abs_ratio << "%)\n";
    // ⚠️ 由绝对阈值比例驱动：过度连接的真实特征是段长绝对超限。
    // MAD 比例仅作信息报告——段长分布天然右偏，MAD 在健康样本上也会标出
    // 长尾的若干百分比（实测正常样本 2RWS 达 4.1%），不宜用作告警触发。
    if (abs_ratio > 1.0) {
        out5 << "⚠️ 注意：骨架可能存在过度连接（绝对超长段比例偏高）\n";
    }
    out5 << "注：仅统计直径 >= 10 μm 且节点数 >= 3 的血管段；"
            "中位数抗离群，离群段仅标记不剔除\n";

    out1.close();
    out2.close();
    out3.close();
    out4.close();
    out5.close();

    return true;
}

}  // namespace vessel_stats
