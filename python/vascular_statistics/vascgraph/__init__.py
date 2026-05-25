"""
VascGraph 子模块垫片。

将 VascGraph 上游包加入 sys.path，使其内部 ``from VascGraph.X import Y``
导入无需修改即可工作。对外暴露统一导入路径。

同时修补 NetworkX 3.x 兼容性：
- G.node[n]  → 在 Graph/DiGraph 中加 @property node 返回 _node
- nx.connected_component_subgraphs → monkey-patch 回 networkx
"""

import sys
import os

# 将 vascgraph/ 父目录加入 sys.path，使 "from VascGraph.X import Y"
# 能正确找到 VascGraph/ 包（而非 VascGraph/VascGraph.py 占位符）。
_parent_dir = os.path.dirname(__file__)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# --- NetworkX 3.x 兼容补丁 ---
import networkx as nx
if not hasattr(nx, "connected_component_subgraphs"):
    def _connected_component_subgraphs(G):
        for c in nx.connected_components(G):
            yield G.subgraph(c).copy()
    nx.connected_component_subgraphs = _connected_component_subgraphs

from VascGraph import GeomGraph as GeomGraph
from VascGraph import Skeletonize as Skeletonize
from VascGraph import GraphIO as GraphIO
from VascGraph import GraphValidation as GraphValidation
from VascGraph import Tools as Tools

try:
    from VascGraph import GraphLab as GraphLab
except ImportError:
    GraphLab = None

try:
    from VascGraph import FlowSimulation as FlowSimulation
except ImportError:
    FlowSimulation = None

try:
    from VascGraph import GraphFlowSimulation as GraphFlowSimulation
except ImportError:
    GraphFlowSimulation = None

try:
    from VascGraph import InitFlowSimulation as InitFlowSimulation
except ImportError:
    InitFlowSimulation = None
