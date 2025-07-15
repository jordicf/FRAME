from frame.netlist.netlist import Netlist
from frame.netlist.netlist_types import HyperEdge, NamedHyperEdge
from collections import defaultdict
from tools.early_router.hanan import (HananGrid, HananEdge3D, HananGraph3D,
                                      Layer, HananNode3D)
from tools.early_router.utils import compute_node_degrees
from tools.early_router.types import NetId, VarId, EdgeID, CellId
from tools.early_router.utils import rescale
import warnings
from typing import Dict, List, Tuple, Any
import math
from typing import List, Tuple, Optional


# STEP 1: Import solver
from ortools.math_opt.python import mathopt
from ortools.math_opt.python.mathopt import Model, Variable, SolveResult, SolveParameters, ModelSolveParameters


class FeedThrough:
    # Check if the 2D graph is a reduction of the 3D graph with one layer 'HV'
    _graph: HananGraph3D
    _solver: Model

    class VariableStore:
        """Class to store the Variables from the model."""

        _vars: dict[EdgeID,
                    dict[NetId, tuple[Variable, tuple[bool, bool, bool]]]]
        _tol: float

        def __init__(self, tol: float = 1e-6):
            # tuple[bool,bool,bool] = [Crossing, Via, Source]
            self._vars = dict()
            self._tol = tol

        def add(self, var_id: VarId, var: Variable, crossing: bool = False,
                via: bool = False, source: bool = False) -> None:
            """Add a variable with a complex key."""
            self._vars.setdefault((var_id[0], var_id[1]), {})[
                var_id[2]] = (var, (crossing, via, source))

        def get_variables(
            self,
            edge_id: Optional[EdgeID] = None,
            net_id: Optional[NetId] = None,
            crossing: Optional[bool] = None,
            via: Optional[bool] = None,
            source: Optional[bool] = None
        ) -> list[Variable]:
            """Retrieve variables stored in the class"""
            if edge_id is not None:
                if edge_id in self._vars:
                    keys = [edge_id]
                else:
                    keys = []
            else:
                keys = self._vars.keys()
            return [
                info_var[0]
                for k in keys
                for ni, info_var in self._vars[k].items()
                if (net_id is None or net_id == ni)
                and (crossing is None or crossing == info_var[1][0])
                and (via is None or via == info_var[1][1])
                and (source is None or source == info_var[1][2])
            ]

        def get_variable(self, key: VarId) -> Variable:
            """Retrieve a variable by its full key."""
            return self._vars.get((key[0], key[1]), {}).get(key[2], [None])[0]

        def update(self, result: SolveResult):
            """Update the routes and congestion properties.
            Useful after solving the problem."""
            vals = result.variable_values()  # Dict var:value
            self._routes: dict[NetId, list[dict[EdgeID, float]]] = {}
            self._congestion: dict[EdgeID, float] = {}
            for var_id, var, flags in iter(self):
                edge_id = (var_id[0], var_id[1])
                net_id = var_id[2]
                wire = vals.get(var, None)
                if wire > self._tol:
                    # Remind, we only saved the "important vars,
                    # not subnets vars", otherwise check 'x' in var.name
                    self._routes.setdefault(net_id, []).append(
                        {edge_id: wire}
                    )
                    reverse_key = edge_id[::-1]
                    self._congestion[edge_id] = \
                        self._congestion.get(edge_id, 0) + \
                        wire + self._congestion.pop(reverse_key, 0)

        def get_unrouted(self, result: SolveResult,
                         nets: dict[NetId, NamedHyperEdge | HyperEdge]) \
                -> list[NetId]:
            """Update the routes and congestion properties.
            Useful after solving the problem."""
            vals = result.variable_values()  # Dict var:value
            unrouted: list[NetId] = []
            for net_id in nets:
                vars = self.get_variables(net_id=net_id, source=True)
                wire_routed = sum(vals[var] for var in vars)
                wei = nets[net_id].weight
                if wire_routed < wei:
                    unrouted.append(net_id)
            return unrouted

        def delete_net(self, net_id: NetId) -> None:
            """Delete all data related to a given NetId from self._vars."""
            for edge_id in list(self._vars.keys()):
                if net_id in self._vars[edge_id]:
                    del self._vars[edge_id][net_id]
                    # Clean up empty sub-dictionaries
                    if not self._vars[edge_id]:
                        del self._vars[edge_id]

        def change_tol(self, new_tol: float):
            """Setter for the new tolerance solution"""
            self._tol = new_tol

        @property
        def routes(self) -> dict[NetId, list[dict[EdgeID, float]]]:
            """Returns a dictionary with the routing solution for each net."""
            if hasattr(self, "_routes"):
                return self._routes

        @property
        def congestion(self) -> dict[EdgeID, float]:
            """Returns a dictionary with the number of wires
            that cross each edge."""
            if hasattr(self, "_congestion"):
                return self._congestion

        def __iter__(self):
            for edge_id, net_dict in self._vars.items():
                for net_id, (var, flags) in net_dict.items():
                    yield ((edge_id[0], edge_id[1], net_id), var, flags)

    def __init__(self, input_data: Netlist | HananGrid, **kwargs):
        """Initialize the FeedThrough class with a Netlist or
        a HananGrid. Extra arguments can be passed for a advanced settings
        such as layers"""

        self._nets: dict[int, NamedHyperEdge] = {}
        self._m: dict[str, list] = {}  # Metrics
        self.reweight = kwargs.get('reweight_nets_range', None)
    
        l = kwargs.get('layers')
        h_pitch, v_pitch = kwargs.get('pitch_layers', (1,1))
        if not l:
            lay = [Layer('H',pitch=h_pitch), Layer('V',pitch=v_pitch)]

        asap7 = kwargs.get('asap7', False) # Uses asap7 tech for 76 nm layers

        if isinstance(input_data, Netlist):
            hanan_grid = HananGrid(input_data)
            self._graph = HananGraph3D(hanan_grid, lay, input_data,
                                       asap7=asap7)
            self._m['n_terminals'] = len(
                [1 for m in input_data.modules if m.is_terminal])
            self._m['n_modules'] = input_data.num_modules - \
                self._m['n_terminals']
        elif isinstance(input_data, HananGrid):
            hanan_grid = input_data
            self._graph = HananGraph3D(hanan_grid, layers=lay)

        # Add some metrics
        self._m['n_layers'] = len(self._graph.layers)
        self._m['n_cells'] = len(hanan_grid.cells)
        self._m['n_nodes'] = len(self._graph.nodes)
        self._m['size'] = hanan_grid.shape

    def add_nets(self, nets: list[HyperEdge | NamedHyperEdge] |
                 dict[NetId, HyperEdge | NamedHyperEdge]) -> None:
        """Add to the class nets to be routed"""
        if isinstance(nets, list):
            it = enumerate(nets, start=max(self._nets)+1 if self._nets else 0)
        else:
            it = nets.items()
        for net_id, net in it:
            assert not (net_id in self._nets), "NetId already exists"
            if isinstance(net, HyperEdge):
                modules = [m.name for m in net.modules]
            else:
                modules = net.modules
            if net.weight < 1:
                warnings.warn(
                    f"Converting netid ({net_id}) weight " \
                    f"lower than 1 to an int!", UserWarning)
                wei = net.weight
            else:
                wei = int(net.weight)
            self._nets[net_id] = NamedHyperEdge(modules, wei)

    def _add_2pin_net(self, net_id: int, net: NamedHyperEdge):
        net_nodes = self._graph.get_net_boundingbox(net, self.nets_bb[net_id])
        net_edges = self._graph.get_edgesid_subset(net_nodes)

        modulenames = net.modules
        wei = net.weight  # wires to connect

        source_edges = self._graph.get_crossings_by_modulename(modulenames[0])
        sink_edges = self._graph.get_crossings_by_modulename(modulenames[1])
        net_variables: dict[VarId, Variable] = {}
        net_constraints: list = []
        # STEP 3: Define the variables
        for e_id in net_edges:
            e = self._graph.get_edge(e_id[0], e_id[1])
            var: Variable = self._solver.add_variable(lb=0, ub=int(
                wei) + 1, name=f'x_{e.source._id}_{e.target._id}_{net_id}')
            net_variables[(e.source._id, e.target._id, net_id)] = var
            s = e in source_edges['out']
            avoidable_crossing = False
            if (e.crossing and
                e.source.modulename not in modulenames and
                e.target.modulename not in modulenames):
                # In a 2-pin net, the unavoidable crossings are the ones
                # that source->target, source or target is in module names
                # making the entering and exiting edges not counted in module
                # crossings minimization.
                avoidable_crossing = True
            self._variables.add((e.source._id, e.target._id, net_id),
                                var, crossing=avoidable_crossing,
                                via=e.via, source=s)

        # STEP 4: State the constraints
        # Adding constraints on flow conservation
        for n in net_nodes:
            # We have to add constraints on all nodes
            # except the source and sink
            if n.modulename in modulenames:
                continue
            in_out_edges: dict[str, list[HananEdge3D]
                               ] = self._graph.get_edges_from_node(n)
            in_var = [net_variables[(e.source._id, e.target._id, net_id)] for e in in_out_edges["in"] if (
                e.source._id, e.target._id) in net_edges]
            out_var = [net_variables[(e.source._id, e.target._id, net_id)] for e in in_out_edges["out"] if (
                e.source._id, e.target._id) in net_edges]
            c = self._solver.add_linear_constraint(
                sum(in_var) - sum(out_var) == 0)
            net_constraints.append(c)

        # Source and Sink are the only nodes that have flow non-zero
        # For source, the outflow is net weigth and the inflow is 0
        c = self._solver.add_linear_constraint(sum(net_variables[(
            e.source._id, e.target._id, net_id)] for e in source_edges['out'] if (e.source._id, e.target._id) in net_edges) <= wei)
        net_constraints.append(c)
        c = self._solver.add_linear_constraint(sum(net_variables[(
            e.source._id, e.target._id, net_id)] for e in source_edges['in'] if (e.source._id, e.target._id) in net_edges) == 0)
        net_constraints.append(c)
        # For sink, the inflow is the net weight and the outflow is 0
        c = self._solver.add_linear_constraint(sum(net_variables[(
            e.source._id, e.target._id, net_id)] for e in sink_edges['in'] if (e.source._id, e.target._id) in net_edges) <= wei)
        net_constraints.append(c)
        c = self._solver.add_linear_constraint(sum(net_variables[(
            e.source._id, e.target._id, net_id)] for e in sink_edges['out'] if (e.source._id, e.target._id) in net_edges) == 0)
        net_constraints.append(c)

        strict = []
        c = sum(net_variables[(e.source._id, e.target._id, net_id)] for e in source_edges['out'] if (
            e.source._id, e.target._id) in net_edges) == wei
        strict.append(c)
        c = sum(net_variables[(e.source._id, e.target._id, net_id)] for e in sink_edges['in'] if (
            e.source._id, e.target._id) in net_edges) == wei
        strict.append(c)

        return list(net_variables.values()), net_constraints, strict

    def _add_multipin_net(self, net_id:int, net: NamedHyperEdge):
        net_nodes = self._graph.get_net_boundingbox(net, self.nets_bb[net_id])
        nodesids = set(n._id for n in net_nodes)
        net_edges = set(self._graph.get_edgesid_subset(net_nodes))

        modulenames = net.modules
        wei = net.weight # wires to connect

        shared_variables: dict[VarId, Variable] ={}

        # Split TODO MST (minimum spanning tree)?
        subnets = [NamedHyperEdge(modulenames[i:2+i], wei)
                   for i in range(len(modulenames)-1)]
        specific_variables: dict[str, dict[VarId, Variable]] = {}
        for i, subnet in enumerate(subnets):
            specific_variables[f"s{i}"] = self._add_sub2pin_net(
                net_id, subnet, f"s{i}", net_nodes, net_edges)

        # STEP 3 & 4: Define the shared one-way variables and state constraints
        visited = set()
        for source in self._graph.adjacent_list:
            if not source in nodesids:
                continue
            visited.add(source)
            for target in self._graph.adjacent_list[source]:
                if not target in nodesids:
                    continue
                if target in visited:
                    # Means shared_variables[(target, source, net_id)] is already created
                    for key in specific_variables:
                        self._solver.add_linear_constraint(
                            specific_variables[key][(source, target, net_id)] <= shared_variables[(
                                target, source, net_id)]
                        )
                    continue
                var: Variable = self._solver.add_variable(
                    lb=0, ub=(int(wei) + 1)*len(subnets), name=f'x_{source}_{target}_{net_id}')
                shared_variables[(source, target, net_id)] = var
                avoidable_crossing = False
                e = self._graph.get_edge(source, target)
                # From both direction we only save one, so just if source/target is in modulenames is unavoidable
                if e.crossing and not (e.source.modulename in modulenames) and not (e.target.modulename in modulenames):
                    avoidable_crossing = True
                self._variables.add((source, target, net_id),
                                    var, crossing=avoidable_crossing, via=e.via)
                for key in specific_variables:
                    self._solver.add_linear_constraint(
                        specific_variables[key][(
                            source, target, net_id)] <= var
                    )
        return [var for key in specific_variables for var in specific_variables[key].values()]

    def _add_sub2pin_net(self, net_id: int, subnet: NamedHyperEdge, subnet_id: str, nodes: list[HananNode3D], edgeids: set[EdgeID]) -> dict[VarId, Variable]:
        modulenames = subnet.modules
        net_variables: dict[VarId, Variable] ={}
        net_constraints: list=[]
        ### STEP 3: Define the variables
        for e_id in edgeids:
            var: Variable = self._solver.add_variable(lb=0, ub=int(
                subnet.weight) + 1, name=f'{subnet_id}_{e_id[0]}_{e_id[1]}_{net_id}')
            net_variables[(e_id[0], e_id[1], net_id)] = var

        # STEP 4: State the constraints
        source_edges = self._graph.get_crossings_by_modulename(modulenames[0])
        sink_edges = self._graph.get_crossings_by_modulename(modulenames[1])

        # Adding constraints on flow conservation
        for n in nodes:
            # We have to add constraints on all nodes except the source and sink
            if n.modulename in modulenames:
                continue
            in_out_edges: dict[str, list[HananEdge3D]] = self._graph.get_edges_from_node(n)
            in_var = [net_variables[(e.source._id, e.target._id, net_id)] for e in in_out_edges["in"] if (e.source._id, e.target._id) in edgeids]
            out_var = [net_variables[(e.source._id, e.target._id, net_id)] for e in in_out_edges["out"] if (e.source._id, e.target._id) in edgeids]
            c = self._solver.add_linear_constraint(sum(in_var) - sum(out_var) == 0)
            net_constraints.append(c)

        # Source and Sink are the only nodes that have flow non-zero
        # For source, the outflow is net weigth and the inflow is 0
        # ALL strict so we ensure that multiple-pin nets are routed  not as in <= in 2-pin nets
        c = self._solver.add_linear_constraint(sum(net_variables[(e.source._id, e.target._id, net_id)] for e in source_edges['out'] if (e.source._id, e.target._id) in edgeids) == subnet.weight)
        net_constraints.append(c)
        c = self._solver.add_linear_constraint(sum(net_variables[(e.source._id, e.target._id, net_id)] for e in source_edges['in']  if (e.source._id, e.target._id) in edgeids) == 0)
        net_constraints.append(c)
        # For sink, the inflow is the net weight and the outflow is 0
        c = self._solver.add_linear_constraint(sum(net_variables[(e.source._id, e.target._id, net_id)] for e in sink_edges['in'] if (e.source._id, e.target._id) in edgeids) == subnet.weight)
        net_constraints.append(c)
        c = self._solver.add_linear_constraint(sum(net_variables[(e.source._id, e.target._id, net_id)] for e in sink_edges['out'] if (e.source._id, e.target._id) in edgeids) == 0)
        net_constraints.append(c)

        # in2pin net the constraints are in net_constraints[-2] and net_constraints[-4]
        return net_variables, net_constraints

    def _add_capacity_constraints(self):
        # STEP 4: Add capacity constraints
        # Add constraints on not exceeding capacity
        capacities: list = []
        visited = set()
        for source_id in self._graph.adjacent_list:
            if not (source_id in visited):
                visited.add(source_id)
                source = self._graph.get_node(source_id)
                if self._graph.is_terminal(source):
                    # The capacity is not set beause it is infinity
                    continue
                for target_id, e in self._graph.adjacent_list[source_id].items():
                    target = self._graph.get_node(target_id)
                    if target_id in visited or self._graph.is_terminal(target):
                        continue
                    vars1 = self._variables.get_variables(
                        edge_id=(source_id, target_id))
                    vars2 = self._variables.get_variables(
                        edge_id=(target_id, source_id))
                    if len(vars1+vars2) == 0:
                        continue
                    c = self._solver.add_linear_constraint(
                        sum(vars1 + vars2) <= e.capacity
                    )  # Capacity set here e.capacity
                    capacities.append(c)
        return capacities

    def set_capacity_adjustments(self, cap_adjust: dict[EdgeID, float | int]):
        self._graph.apply_capacity_adjustments(cap_adjust)

    def _compute_norms(self):
        hpwl = 0
        for n in self._nets.values():
            centers_x = {c.center.x for m in n.modules for c in self._graph.get_nodes_by_modulename(
                m)}  # Should compute the centroid and not the cell
            centers_y = {
                c.center.y for m in n.modules for c in self._graph.get_nodes_by_modulename(m)}
            # Manhattan distance
            if centers_x and centers_y:  # Ensure we don't call max/min on empty lists
                hpwl += (max(centers_x) - min(centers_x) + 
                         max(centers_y) - min(centers_y)) * n.weight
        vu = sum([n.weight for n in self._nets.values()])
        mc = 4 * vu # This is an approximation based on some experiments
        if hasattr(self, 'pre_result'):
            module_crossings = self._variables.get_variables(crossing=True)
            mc = sum(self.pre_result.variable_values(module_crossings))
            via_usage = self._variables.get_variables(via=True)
            vu = sum(self.pre_result.variable_values(via_usage))
            wl = [self.pre_result.variable_values(var) * self._graph.get_edge(var_id[0], var_id[1]).length
                  for var_id, var, _ in self._variables]
            hpwl = sum(wl)
        self.norms = (hpwl, mc, vu)
        return

    def _find_opt_bb(self, unrouted_nets: list[NetId] = None, depth: int = 0, max_depth: int = 3):

        if unrouted_nets is None:  # Solving call
            hpwl = self.norms[0]
            # STEP 5: Define the objective & Compute the normalization factors.
            self._solver.maximize_linear_objective(
                sum(self._variables.get_variables(source=True)) -
                sum(self._variables.get_variables(source=False))/hpwl
            )
            params = SolveParameters()
            params.highs.int_options['user_cost_scale'] = int(math.log2(len(self._nets)))
            params.highs.double_options['primal_feasibility_tolerance'] = 1e-5
            params.highs.double_options['dual_feasibility_tolerance'] = 1e-3
            ### STEP 6: Solve the model
            self.pre_result = mathopt.solve(
                 self._solver, mathopt.SolverType.HIGHS, params=params)
            ### Retrieve results
            unrouted = self._variables.get_unrouted(
                 self.pre_result,self._nets)
            print(f"[Iteration {depth}] Unrouted: {len(unrouted)} nets out of {len(self._nets)} "
                  f"with {self.pre_result.solve_time().total_seconds()} secs")
            if len(unrouted) > 10:
                return False
            self._m.setdefault('#unrouted_nets',[]).append(len(unrouted))
            self._m.setdefault('unrouted_nets_ids',[]).append(unrouted)
            return self._find_opt_bb(unrouted, depth+1)

        elif len(unrouted_nets) == 0:  # Finish call
            # Change the last constraints of the model <= (sources and sinks) to ==.
            for net_id in self._nets:
                net_vars, net_constraints, net_eq_constraints = self.model_components[net_id]
                # In _add_2pin_net, the constraints are in net_constraints[-2] and net_constraints[-4]
                # In multiple pin net...
                self._solver.delete_linear_constraint(net_constraints[-2])
                self._solver.delete_linear_constraint(net_constraints[-4])
                for c in net_eq_constraints:
                    self._solver.add_linear_constraint(c)
            return True
        elif depth > max_depth:  # Finish call
            print("Max recursion depth reached. Exiting.")
            return False
        else:
            # Update BoundingBox and model call
            for net_id in unrouted_nets:
                self.nets_bb[net_id] += 1
                net_vars, net_constraints, net_eq_constraints = self.model_components.pop(
                    net_id, ([], [], []))
                # Deleting from self._variables
                self._variables.delete_net(net_id)
                for var in net_vars:
                    self._solver.delete_variable(var)
                for c in net_constraints:
                    self._solver.delete_linear_constraint(c)
                net = self._nets[net_id]
                if len(net.modules) > 2:
                    self.shared = self._add_multipin_net(net_id, net)
                else:
                    self.model_components[net_id] = self._add_2pin_net(
                        net_id, net)

            model_constraints = self.model_components.pop(-1, [])
            for c in model_constraints:
                self._solver.delete_linear_constraint(c)
            self.model_components[-1] = self._add_capacity_constraints()
            return self._find_opt_bb(None, depth)

    def build(self, optimize_bbox=True, max_depth= 3, infinite_cap = False, bounding_box=True) -> bool:

        self._variables = self.VariableStore()
        if bounding_box:
            self.nets_bb = defaultdict(int)
        else:
            m = max([self._m['size'].w, self._m['size'].h])
            self.nets_bb = defaultdict(lambda:5)
        self._solver = mathopt.Model()

        if self.reweight:
            weigths = [net.weight for net in self._nets.values()]
            low, high = self.reweight
            print(
                f"Rescaling net weights to be in the interval ({low}, {high})")
            old_min = min(weigths)
            old_max = max(weigths)
            for net in self._nets.values():
                new_w = rescale(net.weight, old_min, old_max)
                net.weight = int(new_w)

        # Add to the model the varaibles and constraints
        self.model_components: dict[NetId] = {}
        for net_id, net in self._nets.items():
            if len(net.modules)>2:
                # TODO
                self._add_multipin_net(net_id, net)
            else:
                self.model_components[net_id] = self._add_2pin_net(net_id, net)

        if infinite_cap:
            self._compute_norms()
            self.is_build = True
            for net_id in self._nets:
                net_vars, net_constraints, net_eq_constraints = self.model_components[net_id]
                # In _add_2pin_net, the constraints are in net_constraints[-2] and net_constraints[-4]
                self._solver.delete_linear_constraint(net_constraints[-2])
                self._solver.delete_linear_constraint(net_constraints[-4])
                for c in net_eq_constraints:
                    self._solver.add_linear_constraint(c)
            return True
        self.model_components[-1] = self._add_capacity_constraints()

        self._compute_norms()
        if bounding_box and optimize_bbox:
            if not self._find_opt_bb(max_depth=max_depth):
                self.is_build = False
                return False
            self._compute_norms()
            self.is_build = True
        return True

    def solve(self, f_wl: float = 0.1, f_mc: float = 0.2, f_vu: float = 0.7) -> tuple[bool, dict[str, int | float]]:
        """
        :return is_solved (bool): Whether the solver found a solution or not
        :return metrics (dict[str,int|float]): Execution metrics and solution values
        """
        assert hasattr(self, 'is_build'), "Before solving, build the model"
        assert self.is_build, "The model has failed when building, change parameters before solving"
        
        self._m['factors'] = (round(f_wl,3),round(f_mc,3),round(f_vu,3))
        self._m['norm_factwl'] = self.norms[0]
        self._m['norm_factmc'] = self.norms[1]
        self._m['norm_factvu'] = self.norms[2]
        
        ### STEP 5: Define the objective
        module_crossings = self._variables.get_variables(crossing=True)
        wire_length = [var * self._graph.get_edge(var_id[0], var_id[1]).length
                       for var_id, var, _ in self._variables]
        via_usage = self._variables.get_variables(via=True)

        self._solver.minimize_linear_objective(
            f_wl * sum(wire_length)/self.norms[0] +
            f_mc * sum(module_crossings)/self.norms[1] +
            f_vu * sum(via_usage)/self.norms[2]
        )
        # Change params
        params = SolveParameters(enable_output=True)
        params.highs.int_options['user_cost_scale'] = int(math.log2(len(self._nets)))
        # params.highs.string_options['solver'] = "ipm"
        # params.highs.string_options['solver'] = "simplex" #(default)
        
        #params.highs.double_options['primal_feasibility_tolerance'] = # Looser constraints (faster, but less accurate)
        #params.highs.double_options['dual_feasibility_tolerance'] = # May accept suboptimal solutions faster
        #params.highs.double_options['ipm_optimality_tolerance'] = # Only for Interior Point Method
        
        # Solve with these parameters
        status = mathopt.solve(
            self._solver,
            mathopt.SolverType.HIGHS,
            params=params
        )
        self.execution_time = status.solve_time().total_seconds()
        self.is_solved = self.has_solution(status)
        self.result = status
        self._variables.update(status)
        self._metrics()
        return self.is_solved, self.metrics

    @property
    def solution(self) -> VariableStore:
        if not self.is_solved:
            warnings.warn(
                "No solution exists for the given problem!", UserWarning)
        return self._variables

    @property
    def hanan_graph(self) -> HananGraph3D:
        return self._graph

    @property
    def metrics(self) -> dict[str, int | float]:
        return self._m

    def _metrics(self):
        self._m['n_nets'] = len(self._nets)
        self._m['variables_k'] = int(len(list(self._solver.variables()))/1000)
        self._m['constraints_k'] = int(
            len(list(self._solver.linear_constraints()))/1000)
        self._m['elapsed_time_secs'] = round(self.execution_time, 3)
        if self.is_solved:
            wire_length = [self.result.variable_values(var) * self._graph.get_edge(
                var_id[0], var_id[1]).length for var_id, var, _ in self.solution]
            self._m['total_wl'] = round(sum(wire_length), 0)
            self._m['module_crossings'] = round(
                sum(self.result.variable_values(self.solution.get_variables(crossing=True))), 1)
            self._m['via_usage'] = round(
                sum(self.result.variable_values(self.solution.get_variables(via=True))), 1)
            self._m['n_bifurations'] = len([1 for netid, route in self.solution.routes.items(
            ) if self.has_bifurcation(self._nets[netid], route)])
            self._m['obj_val'] = round(self.result.objective_value(), 3)
            self._m['int_violations'] = self.check_integral_solution()[1]
            self._m['mrd'] = max(
                (v[0] / v[1]) if v[1] != 0 else 0
                for v in self._congestion_map().values()
            )  # maximum routing density(MRD)

    def has_bifurcation(self, net: NamedHyperEdge, route: list[dict[EdgeID, float]]) -> bool:
        """
        Checks whether a route solution has a bifurcation (nodes with degree > 2 or source/sink nodes with degree > 1)

        :param route: a list of dict[EdgeId, float] with the float as the number of wires through edge (key).
        :param net: a HyperEdge with the net information

        :return bool:
        """
        # If any node has degree higher than 2, implies a bifurcation
        degrees = compute_node_degrees(route)
        for n, d in degrees.items():
            if d > 2:
                return True
        # Check if the source/sink module has degree > 1 (if has two entry/exit implies bifurcation)
        nodes_set = [set([n._id for n in self._graph.get_nodes_by_modulename(m)])
                     for m in net.modules]
        for nodes in nodes_set:
            d = len([n for n in nodes if n in degrees])
            if d > 1:
                return True
        return False

    def save(self, netnames: dict[NetId, str] = {}, filepath="./", filename='routes', extension='.txt'):

        with open(f"{filepath}{filename}{extension}", 'w') as f:
            for netid, route in self.solution.routes.items():
                f.write(
                    f"{netnames.get(netid, 'n')} {netid} {len(route)} {self._nets[netid].weight}\n")
                # for r in self.get_route_path(route):
                for r in route:
                    f.write(f"{list(r.keys())[0][0]}-{list(r.keys())[0][1]}\n")
                f.write("!\n")

    def has_solution(self, status: SolveResult | None = None) -> bool:
        if status is None:
            if hasattr(self, 'is_solved'):
                return self.is_solved
            else:
                return None
        msg = status.termination.detail
        if status.termination.reason == mathopt.TerminationReason.OPTIMAL:
            print(
                f"A probably optimal solution (up to numerical tolerances) has been found.\n{msg}")
            return True
        elif status.termination.reason == mathopt.TerminationReason.FEASIBLE:
            # See SolveResultProto.limit_detail for detailed description of the kind of limit that was reached.
            print(
                f"The optimizer reached some kind of limit and a primal feasible solution is returned.")
            return True
        elif status.termination.reason == mathopt.TerminationReason.INFEASIBLE:
            print(f"The primal problem has no feasible solutions.\n{msg}")
            return False
        elif status.termination.reason == mathopt.TerminationReason.UNBOUNDED:
            print(
                f"The primal problem is feasible and arbitrarily good solutions can be found along a primal ray.\n{msg}")
            return False
        elif status.termination.reason == mathopt.TerminationReason.INFEASIBLE_OR_UNBOUNDED:
            # status.solve_stats.to_proto().problem_status
            print(
                f"The primal problem is either infeasible or unbounded.\n{msg}")
            return False
        elif status.termination.reason == mathopt.TerminationReason.IMPRECISE:
            print("The problem was solved to one of the criteria above (Optimal, Infeasible, Unbounded, or InfeasibleOrUnbounded), ",
                  "but one or more tolerances was not met. Users can still query primal/dual solutions/rays and solution stats, but they",
                  f" are responsible for dealing with the numerical imprecision.\n{msg}")
            return False
        elif status.termination.reason == mathopt.TerminationReason.NO_SOLUTION_FOUND:
            # See SolveResultProto.limit_detail for detailed description of the kind of limit that was reached.
            print(
                f"The optimizer reached some kind of limit and it did not find a primal feasible solution.\n{msg}")
            return False
        elif status.termination.reason == mathopt.TerminationReason.NUMERICAL_ERROR:
            print(
                f"The algorithm stopped because it encountered unrecoverable numerical error. No solution information is present.\n{msg}")
            return False
        elif status.termination.reason == mathopt.TerminationReason.OTHER_ERROR:
            print(
                f"The algorithm stopped because of an error not covered by one of the statuses defined above. No solution information is present.\n{msg}")
            return False
        else:
            print("Unknown status returned!")
            return False

    def get_route_path(self, identifiers: list[dict[EdgeID, float]]) -> list[dict[EdgeID, float]]:
        """return path for a route"""
        # Step 1: Build graph and in-degree map
        graph = {}
        in_degree = defaultdict(int)
        edge_weights = {
            list(edge_dict.keys())[0]: list(edge_dict.values())[0]
            for edge_dict in identifiers
        }

        for identifier in identifiers:
            if len(identifier) != 1:
                raise ValueError(
                    "Each identifier must contain exactly one edge.")
            edge = list(identifier.items())[0]  # ((from, to), weight)
            fromid, toid = edge[0]
            graph[fromid] = toid
            in_degree[toid] += 1
            in_degree[fromid] += 0  # ensure it's in the map

        # Step 2: Find the start node (in-degree == 0)
        start_nodes = [node for node in in_degree if in_degree[node] == 0]
        if not start_nodes:
            return None  # No valid start, maybe a cycle
        start_node = start_nodes[0]

        # Step 3: Walk the path
        path = []
        current = start_node
        visited = set()

        while current in graph:
            next_node = graph[current]
            path.append(
                {(current, next_node): edge_weights[(current, next_node)]})
            if current in visited:
                return None  # Cycle detected
            visited.add(current)
            current = next_node

        return path if len(path) == len(identifiers) else None

    def _congestion_map(self, layer_id: list[int] = []) -> dict[tuple[CellId, CellId], tuple[float, float]]:

        edge_map: dict[tuple[CellId, CellId], tuple[float, float]] = {}
        for edge_id, congestion in self.solution.congestion.items():
            if congestion < 1:
                continue
            edge = self.hanan_graph.get_edge(edge_id[0], edge_id[1])
            from_node = self.hanan_graph.get_node(edge_id[0])
            to_node = self.hanan_graph.get_node(edge_id[1])
            # Skip vias
            if from_node._id[-1] != to_node._id[-1]:
                continue
            elif layer_id and from_node._id[-1] in layer_id:
                # Skip non-layer selecetd
                continue

            sum_con, sum_cap = edge_map.get(
                (from_node._id[:2], to_node._id[:2]), (0, 0))
            sum_con += congestion
            sum_cap += edge.capacity
            edge_map[(from_node._id[:2], to_node._id[:2])] = (sum_con, sum_cap)

        return edge_map

    def check_integral_solution(self, tol: float = 1e-9) -> Tuple[bool, List[Tuple[NetId, EdgeID, float]]]:
        """
        Check if all route values are integer within a tolerance.

        Args:
            tol: Tolerance for floating point comparison.

        Returns:
            A tuple (is_integral, violations):
                - is_integral (bool): True if all values are integer within tolerance.
                - violations (list): List of (NetId, EdgeID, value) for non-integer values.
        """
        routes = self.solution.routes
        violations = []
        for net_id, edge_dicts in routes.items():
            for edge_dict in edge_dicts:
                for edge_id, value in edge_dict.items():
                    if abs(value - round(value)) > tol:
                        violations.append((net_id, edge_id, value))
        return len(violations) == 0, violations
