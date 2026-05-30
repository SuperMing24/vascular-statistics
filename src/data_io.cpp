#include "vascular_statistics/data_io.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

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

    // VerticesInfo[i] = {x, y, z, radius}
    std::vector<std::vector<double>> vertices_info(idx_max,
                                                   std::vector<double>(4, 0.0));

    double f;
    int count = 0;
    while (in >> f) {
        int idx = count / 6;
        int col = count % 6;
        if (col == 2)       // x
            vertices_info[idx][0] = f;
        else if (col == 3)  // y
            vertices_info[idx][1] = f;
        else if (col == 4)  // z
            vertices_info[idx][2] = f;
        else if (col == 5)  // radius
            vertices_info[idx][3] = f;
        count++;
    }
    in.close();

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
    int no_effective_seg = 0;
    double avg_seg_radius = 0.0;
    double avg_seg_length = 0.0;
    double avg_seg_tortuosity = 0.0;

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
                        out4 << path_length / path_direct_distance << "\n";

                        if (k > 2 && avg_temp >= 2.5) {
                            no_effective_seg++;
                            avg_seg_radius -= (avg_seg_radius - avg_temp) / no_effective_seg;
                            avg_seg_length -= (avg_seg_length - path_length) / no_effective_seg;
                            avg_seg_tortuosity -=
                                (avg_seg_tortuosity - path_length / path_direct_distance) / no_effective_seg;
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
                        out4 << path_length / path_direct_distance << "\n";

                        if (k > 2 && avg_temp >= 2.5) {
                            no_effective_seg++;
                            avg_seg_radius -= (avg_seg_radius - avg_temp) / no_effective_seg;
                            avg_seg_length -= (avg_seg_length - path_length) / no_effective_seg;
                            avg_seg_tortuosity -=
                                (avg_seg_tortuosity - path_length / path_direct_distance) / no_effective_seg;
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

    // ---------- 写汇总文件 ----------
    out5 << "平均直径 (μm): " << avg_seg_radius * 4 << "\n"
         << "平均长度 (μm): " << avg_seg_length * 2 << "\n"
         << "段密度 (seg/mm³): " << static_cast<double>(no_effective_seg) / volume << "\n"
         << "平均弯曲度: " << avg_seg_tortuosity << "\n"
         << "注：仅统计直径 >= 10 μm 且节点数 >= 3 的血管段\n";

    out1.close();
    out2.close();
    out3.close();
    out4.close();
    out5.close();

    return true;
}

}  // namespace vessel_stats
