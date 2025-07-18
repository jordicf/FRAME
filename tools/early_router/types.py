"""Type Alias for the Routing module in FRAME"""

from typing import Tuple

# Cell identifier as a tuple of two integers representing cell position
CellId = Tuple[int, int]

# Node identifier as a tuple of three integers representing cell position and layer
NodeId = Tuple[int, int, int]

# Net identifier as an integer
NetId = int

# Variable identifier
# Format: NodeId from the wires will start, NodeId where the wires will end, for the net i
VarId = Tuple[NodeId, NodeId, NetId]

# Edge identifier as a tuple of two nodes ids
# Format: The source node id, the target node id
EdgeId = Tuple[NodeId, NodeId]
