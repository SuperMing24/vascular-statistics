#include <iostream>
#include <string>
#include <cstdlib>
#include "vascular_statistics/data_io.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "用法: vessel_stats <边表文件> <节点表文件> [组织体积] [sx sy sz]\n"
                  << "  边表文件   - 空白分隔整数对，不含 .txt 后缀\n"
                  << "  节点表文件 - 每节点 6 值 [idx type x y z r]，不含 .txt 后缀\n"
                  << "  组织体积   - 体积 mm^3，默认 0.078\n"
                  << "  sx sy sz   - 体素 spacing (μm/体素)，可选；三者齐备→各向异性\n"
                  << "               物理单位口径，缺省→legacy 各向同性(×2/×4)\n";
        return 1;
    }

    std::string edges_file = argv[1];
    std::string vertices_file = argv[2];
    double volume = (argc >= 4) ? std::stod(argv[3]) : 0.078;

    // 可选体素 spacing：需 sx sy sz 三者齐备（argc>=7），否则走 legacy。
    double sx = (argc >= 7) ? std::stod(argv[4]) : -1.0;
    double sy = (argc >= 7) ? std::stod(argv[5]) : -1.0;
    double sz = (argc >= 7) ? std::stod(argv[6]) : -1.0;

    if (!vessel_stats::GenerateStatistics(edges_file, vertices_file, volume,
                                          sx, sy, sz)) {
        std::cerr << "统计生成失败。\n";
        return 1;
    }

    return 0;
}
