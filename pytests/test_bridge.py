"""bridge.py 格式转换模块的单元测试。"""

import os
import tempfile

from vascular_statistics.bridge import pajek_to_cpp_input, _parse_pos


class TestParsePos:
    def test_standard_pajek_pos(self):
        x, y, z = _parse_pos('"[3.15 73.53 83.22]"')
        assert abs(x - 3.15) < 0.01
        assert abs(y - 73.53) < 0.01
        assert abs(z - 83.22) < 0.01

    def test_no_quotes(self):
        x, y, z = _parse_pos("[1.0 2.0 3.0]")
        assert x == 1.0 and y == 2.0 and z == 3.0

    def test_plain_numbers(self):
        x, y, z = _parse_pos("5.0 6.0 7.0")
        assert x == 5.0 and y == 6.0 and z == 7.0

    def test_2d_fallback(self):
        x, y, z = _parse_pos("1.0 2.0")
        assert x == 1.0 and y == 2.0 and z == 0.0

    def test_empty_gives_zero(self):
        x, y, z = _parse_pos("")
        assert x == 0.0 and y == 0.0 and z == 0.0


class TestPajekToCppInput:
    def test_simple_pajek(self):
        """用最小 Pajek 内容验证转换输出格式。"""
        pajek_content = """*vertices 3
1 "1" 0.0 0.0 ellipse pos "[0.0 0.0 0.0]" r 2.0
2 "2" 0.0 0.0 ellipse pos "[1.0 0.0 0.0]" r 1.5
3 "3" 0.0 0.0 ellipse pos "[2.0 0.0 0.0]" r 1.0
*edges
1 2 1.0
2 3 1.0
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pajek_path = os.path.join(tmpdir, "test.pajek")
            edges_out = os.path.join(tmpdir, "test_edges.txt")
            vertices_out = os.path.join(tmpdir, "test_vertices.txt")

            with open(pajek_path, "w") as f:
                f.write(pajek_content)

            e_path, v_path = pajek_to_cpp_input(pajek_path, edges_out, vertices_out)

            # 验证节点文件格式
            with open(v_path) as f:
                lines = f.readlines()
            assert len(lines) == 3
            parts = lines[0].split()
            assert len(parts) == 6  # idx type x y z r
            assert parts[0] == "1"  # node id

            # 验证边文件格式
            with open(e_path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            parts = lines[0].split()
            assert len(parts) == 2  # n1 n2
