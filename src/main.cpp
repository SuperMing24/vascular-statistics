#include <iostream>
#include <string>
#include <cstdlib>
#include "vascular_statistics/data_io.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "用法: vessel_stats <边表文件> <节点表文件> [组织体积]\n"
                  << "  边表文件   - 空白分隔整数对，不含 .txt 后缀\n"
                  << "  节点表文件 - 每节点 6 值 [idx type x y z r]，不含 .txt 后缀\n"
                  << "  组织体积   - 体积 mm^3，默认 0.078\n";
        return 1;
    }

    std::string edges_file = argv[1];
    std::string vertices_file = argv[2];
    double volume = (argc >= 4) ? std::stod(argv[3]) : 0.078;

    if (!vessel_stats::GenerateStatistics(edges_file, vertices_file, volume)) {
        std::cerr << "统计生成失败。\n";
        return 1;
    }

    return 0;
}
