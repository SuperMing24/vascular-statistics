"""
Pajek / SWC 格式到 C++ 统计代码平面格式的转换桥。

C++ 代码期望的输入格式：
  *_edges.txt    — 空白分隔整数对，无文件头
  *_vertices.txt — 每节点一行：idx type x y z radius

VascGraph 输出的 Pajek 格式：
  *vertices N
  1 "label" 0.0 0.0 ellipse pos "[x y z]" r value
  ...
  *edges
  src dst weight
"""

import re
from typing import Dict, Optional, Tuple

import networkx as nx


def _parse_pos(pos_str: str) -> Tuple[float, float, float]:
    """解析 Pajek pos 属性字符串 '"[x y z]"' → (x, y, z)。"""
    # 去掉引号和方括号
    cleaned = pos_str.strip().strip('"').strip("[").strip("]")
    parts = cleaned.split()
    if len(parts) >= 3:
        return float(parts[0]), float(parts[1]), float(parts[2])
    if len(parts) == 2:
        return float(parts[0]), float(parts[1]), 0.0
    if len(parts) == 1:
        return float(parts[0]), 0.0, 0.0
    return 0.0, 0.0, 0.0


def _extract_node_attrs(graph: nx.Graph) -> Dict[int, dict]:
    """从 networkx 图中提取每个节点的 pos / r / type 属性。

    Pajek 节点属性通过 nx.read_pajek 解析后可能是字符串或数字。
    此函数统一处理这些情况。
    """
    attrs: Dict[int, dict] = {}
    for node_id in graph.nodes():
        data = graph.nodes[node_id]
        result: dict = {"pos": (0.0, 0.0, 0.0), "r": 1.0, "type": 0}

        # 解析 pos
        raw_pos = data.get("pos", data.get("Pos", ""))
        if isinstance(raw_pos, str) and raw_pos:
            result["pos"] = _parse_pos(raw_pos)
        elif isinstance(raw_pos, (list, tuple)) and len(raw_pos) >= 3:
            result["pos"] = (float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2]))

        # 解析 radius（可能是 'r' 或 'd' 属性）
        for r_key in ("r", "d", "radius"):
            if r_key in data:
                try:
                    result["r"] = float(data[r_key])
                except (ValueError, TypeError):
                    pass
                break

        # 解析 type
        if "type" in data:
            try:
                result["type"] = int(float(data["type"]))
            except (ValueError, TypeError):
                pass

        # 转换 node_id 为整数（networkx 读 Pajek 可能是字符串键）
        try:
            int_id = int(node_id)
        except (ValueError, TypeError):
            int_id = hash(node_id) % 10**7  # 极端回退
        attrs[int_id] = result

    return attrs


def pajek_to_cpp_input(
    pajek_path: str,
    edges_out: Optional[str] = None,
    vertices_out: Optional[str] = None,
) -> Tuple[str, str]:
    """将 Pajek 图文件转换为 C++ 统计代码所需的平面格式。

    Args:
        pajek_path: 输入 Pajek .net 文件路径。
        edges_out: 输出边文件路径（默认：<stem>_edges.txt）。
        vertices_out: 输出节点文件路径（默认：<stem>_vertices.txt）。

    Returns:
        (edges_path, vertices_path) — 两个输出文件的路径。
    """
    import os

    stem = os.path.splitext(pajek_path)[0]
    if edges_out is None:
        edges_out = stem + "_edges.txt"
    if vertices_out is None:
        vertices_out = stem + "_vertices.txt"

    g = nx.read_pajek(pajek_path)
    attrs = _extract_node_attrs(g)

    # 写入节点文件：每行 idx type x y z radius
    with open(vertices_out, "w", encoding="utf-8") as f:
        for nid in sorted(attrs.keys()):
            a = attrs[nid]
            f.write(f"{nid} {a['type']} {a['pos'][0]:.6f} {a['pos'][1]:.6f} "
                    f"{a['pos'][2]:.6f} {a['r']:.6f}\n")

    # 写入边文件：每行 n1 n2
    with open(edges_out, "w", encoding="utf-8") as f:
        for u, v in g.edges():
            try:
                u_int = int(u)
                v_int = int(v)
            except (ValueError, TypeError):
                u_int = u
                v_int = v
            f.write(f"{u_int} {v_int}\n")

    return edges_out, vertices_out


def swc_to_cpp_input(
    swc_path: str,
    edges_out: Optional[str] = None,
    vertices_out: Optional[str] = None,
) -> Tuple[str, str]:
    """将 SWC 文件转换为 C++ 统计代码的平面格式。

    SWC 格式：每行 n type x y z r parent
    仅转换父子关系为边。

    Args:
        swc_path: 输入 SWC 文件路径。
        edges_out: 输出边文件路径。
        vertices_out: 输出节点文件路径。

    Returns:
        (edges_path, vertices_path)。
    """
    import os

    stem = os.path.splitext(swc_path)[0]
    if edges_out is None:
        edges_out = stem + "_edges.txt"
    if vertices_out is None:
        vertices_out = stem + "_vertices.txt"

    vertices: Dict[int, Tuple[float, float, float, float, int]] = {}
    edges: list = []

    with open(swc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            nid = int(parts[0])
            ntype = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            r = float(parts[5])
            parent = int(parts[6])
            vertices[nid] = (x, y, z, r, ntype)
            if parent > 0:
                edges.append((parent, nid))

    with open(vertices_out, "w", encoding="utf-8") as f:
        for nid in sorted(vertices.keys()):
            x, y, z, r, ntype = vertices[nid]
            f.write(f"{nid} {ntype} {x:.6f} {y:.6f} {z:.6f} {r:.6f}\n")

    with open(edges_out, "w", encoding="utf-8") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")

    return edges_out, vertices_out
