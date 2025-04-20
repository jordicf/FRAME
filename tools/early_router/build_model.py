from frame.netlist.netlist import Netlist
from frame.geometry.geometry import BoundingBox, Point
from frame.netlist.netlist_types import HyperEdge, NamedHyperEdge
from collections import defaultdict
from tools.early_router.hanan import HananGrid, HananEdge3D, HananGraph3D, Layer, HananNode3D
from tools.early_router.utils import kruskal, compute_node_degrees
from tools.early_router.types import NetId, VarId, EdgeID, NodeId
import warnings


### STEP 1: Import solver
from ortools.math_opt.python import mathopt
from ortools.math_opt.python.mathopt import Model, Variable, SolveResult


class FeedThrough:
    _graph: HananGraph3D # Check if the 2D graph is a reduction of the 3D graph with one layer 'HV'
    _solver: Model


    class VariableStore:
        """Class to store the Variables from the model."""
        def __init__(self, tol=1e-6):
            self._wl: dict[VarId,Variable] = {}
            self._crossings:list[Variable] = []
            self._vias:list[Variable] = []
            self._tol = tol

        def add(self, key: VarId, var:Variable, crossing=False, via = False):
            """Add a variable with a complex key."""
            assert not(key in self._wl), "Key already used"
            self._wl[key] = var
            if crossing:
                self._crossings.append(var)
            if via:
                self._vias.append(var)
            
        def get_all(self)->dict[VarId,Variable]:
            """Retrieve all variables stored in the class."""
            return self._wl

        def get_by_edge_id(self, edge_id: EdgeID, num_nets:int)-> list[Variable]:
            """Retrieve all variables with a specific (tuple, tuple) key."""
            variables =[]
            for net_id in range(num_nets):
                key = (edge_id[0], edge_id[1], net_id)
                if key in self._wl:
                    variables.append(self._wl[key])
            return variables

        def get_by_net_id(self, net_id:NetId)->list[Variable]:
            """Retrieve all variables with a specific int key."""
            variables =[]
            for key in self._wl:
                if net_id == key[2]:
                    #variables.append({(key[0],key[1]):self._wl[key]})
                    variables.append(self._wl[key])
            return variables

        def get_varaible(self, key:VarId)->Variable:
            """Retrieve a variable by its full key."""
            return self._wl.get(key, None)

        def delete(self, key:VarId):
            """Delete a variable by its full key."""
            var = self._wl.pop(key, None)
            if var in self._crossings:
                self._crossings.remove(var)
            if var in self._vias:
                self._vias.remove(var)

        def update(self, result: SolveResult):
            """Update the routes and congestion properties. Useful after solving the problem."""
            vals = result.variable_values() # Dict var:value
            self._routes: dict[NetId, list[dict[EdgeID, float]]] = {}
            self._congestion: dict[EdgeID, float] = {}
            for key, var in self._wl.items():
                if var in vals:
                    wire = vals[var]
                    if wire > self._tol and 'x' in var.name:
                        edge_id = (key[0],key[1])
                        self._routes.setdefault(key[2],[]).append(
                            {edge_id: wire}
                        )
                        reverse_key = edge_id[::-1]
                        self._congestion[edge_id] = self._congestion.get(edge_id, 0) + wire + self._congestion.pop(reverse_key, 0)
                    
        def change_tol(self, new_tol:float):
            """Setter for the new tolerance solution"""
            self._tol = new_tol

        @property
        def routes(self)->dict[NetId, list[dict[EdgeID, float]]]:
            """Returns a dictionary with the routing solution for each net."""
            if hasattr(self, "_routes"):
                return self._routes
        
        @property
        def congestion(self)-> dict[EdgeID, float]:
            """Returns a dictionary with the number of cables that crosses each edge."""
            if hasattr(self, "_congestion"):
                return self._congestion

        @property
        def crossings(self)->list[Variable]:
            """Returns the list of varaibles that models the crossings"""
            return self._crossings
        
        @property
        def vias(self)->list[Variable]:
            """Returns the list of varaibles that models the vias"""
            return self._vias
         

    def __init__(self, input_data: Netlist|HananGrid, **kwargs):

        # Consider more arguments, layers list,...
        self._nets: dict[int,HyperEdge|NamedHyperEdge] = {}
        self._m: dict[str, list] = {} # Metrics
        self._norms:list[float,float,float]=[1,1,1] # Normalization factors
    
        ### STEP 2: Declare the solver SCIP (mixed ILP), GLOP (LP), CBC
        self._solver = mathopt.Model()
        if not self._solver:
            print("Solver not available!")
            return
        if kwargs.get('logging',False): p=mathopt.SolveParameters(enable_output=True)
        
        l = kwargs.get('layers',[Layer('H',pitch=76), Layer('V',pitch=76)])

        if isinstance(input_data, Netlist):
            hanan_grid = HananGrid(input_data)
            #self._graph = HananGraph3D(hanan_grid, input_data, reweight_nets=True, normalize_capacities=True)
            self._graph = HananGraph3D(hanan_grid, l, input_data)
            self._m['n_terminals'] = len([1 for m in input_data.modules if m.is_terminal])
            self._m['n_modules'] = input_data.num_modules - self._m['n_terminals']
        elif isinstance(input_data, HananGrid):
            hanan_grid = input_data
            self._graph = HananGraph3D(hanan_grid, layers=l)

        self._variables = self.VariableStore(kwargs.get('solution_tolerance',1e-6))
        # Add some metrics
        self._m['n_layers']=len(self._graph.layers)
        self._m['n_cells'] = len(hanan_grid.cells)
        self._m['n_nodes'] = len(self._graph.nodes)
        self._m['size'] = hanan_grid.shape


    def add_nets(self, nets:list[HyperEdge|NamedHyperEdge]|dict[NetId, HyperEdge|NamedHyperEdge]):
        if isinstance(nets, list):
            for net_id, net in enumerate(nets, start=max(self._nets)+1 if self._nets else 0):
                self._nets[net_id] = net
                if len(net.modules)>2:
                    self.shared = self._add_multipin_net(net_id, net)
                else:
                    self._add_2pin_net(net_id, net)
        else:
            for net_id, net in nets.items():
                assert not(net_id in self._nets), "NetId already exists"
                self._nets[net_id] = net
                if len(net.modules)>2:
                    self.shared = self._add_multipin_net(net_id, net)
                else:
                    self._add_2pin_net(net_id, net)

    def _add_2pin_net(self, net_id:int, net: HyperEdge | NamedHyperEdge):
        net_nodes = self._graph.get_net_boundingbox(net, 2) ######################### CHANGE HERE
        net_edges = set(self._graph.get_edgesid_subset(net_nodes))
        
        if isinstance(net, HyperEdge):
            modulenames = [m.name for m in net.modules]
        elif isinstance(net, NamedHyperEdge):
            modulenames = net.modules
        else:
            assert False, "Net is neither HyperEdge not NamedHyperEdge"
        assert len(modulenames) == 2, "Adding a multiple pin net as a 2-pin net!"
        wei = net.weight # wires to connect
        net_variables: dict[VarId, Variable] ={}
        ### STEP 3: Define the variables
        for e_id in net_edges:
            e = self._graph.get_edge(e_id[0], e_id[1])
            var:Variable = self._solver.add_variable(lb = 0, ub = int(wei) + 1, name= f'x_{e.source._id}_{e.target._id}_{net_id}')
            net_variables[(e.source._id, e.target._id, net_id)] = var
            avoidable_crossing = False
            if e.crossing and not(e.source.modulename in modulenames) and not(e.target.modulename in modulenames):
                # In a 2-pin net, the unavoidable crossings are the ones that source->target, source or target is in module names making 
                # the entering and exiting edges not counted in module crossings minimization.
                avoidable_crossing = True
            self._variables.add((e.source._id, e.target._id, net_id), var, crossing=avoidable_crossing,via=e.via)
    
        ### STEP 4: State the constraints TODO this could be faster by just playing with edge ids -> sets not lists
        source_edges = self._graph.get_crossings_by_modulename(modulenames[0])
        sink_edges = self._graph.get_crossings_by_modulename(modulenames[1])
        # Source and Sink are the only nodes that have flow non-zero
        # For source, the outflow is net weigth and the inflow is 0
        self._solver.add_linear_constraint(sum(net_variables[(e.source._id, e.target._id, net_id)] for e in source_edges['out'] if (e.source._id, e.target._id) in net_edges) == wei)
        self._solver.add_linear_constraint(sum(net_variables[(e.source._id, e.target._id, net_id)] for e in source_edges['in'] if (e.source._id, e.target._id) in net_edges) == 0)
        # For sink, the inflow is the net weight and the outflow is 0
        self._solver.add_linear_constraint(sum(net_variables[(e.source._id, e.target._id, net_id)] for e in sink_edges['in'] if (e.source._id, e.target._id) in net_edges) == wei)
        self._solver.add_linear_constraint(sum(net_variables[(e.source._id, e.target._id, net_id)] for e in sink_edges['out'] if (e.source._id, e.target._id) in net_edges) == 0)

        # Adding constraints on flow conservation
        for n in net_nodes:
            # We have to add constraints on all nodes except the source and sink
            if n.modulename in modulenames:
                continue
            in_out_edges: dict[str, list[HananEdge3D]] = self._graph.get_edges_from_node(n)
            in_var = [net_variables[(e.source._id, e.target._id, net_id)] for e in in_out_edges["in"] if (e.source._id, e.target._id) in net_edges]
            out_var = [net_variables[(e.source._id, e.target._id, net_id)] for e in in_out_edges["out"] if (e.source._id, e.target._id) in net_edges]
            self._solver.add_linear_constraint(sum(in_var) - sum(out_var) == 0)

    def _add_multipin_net(self, net_id:int, net: HyperEdge):

        modulenames = [m.name for m in net.modules]
        wei = net.weight # wires to connect

        net_nodes = self._graph.get_net_boundingbox(net, 0) ######################### CHANGE HERE
        nodesids = set(n._id for n in net_nodes)
        net_edges = set(self._graph.get_edgesid_subset(net_nodes))

        shared_variables: dict[VarId, Variable] ={}

        # Split TODO MST (minimum spanning tree)? 
        subnets = [NamedHyperEdge(modulenames[i:2+i],wei) for i in range(len(modulenames)-1)]
        specific_variables: dict[str,dict[VarId, Variable]] = {}
        for i, subnet in enumerate(subnets):
            specific_variables[f"s{i}"] = self._add_sub2pin_net(net_id,subnet, f"s{i}",net_nodes, net_edges)
        
        ### Add extra constraints for intermediate modules TODO I don't think this is necessary, only incerases the computational cost
        # for i, m in enumerate(modulenames[1:-1]):
        #     extra_var = self._add_intermodule_constraints(net_id,m,wei,
        #         (specific_variables[f"s{i}"],specific_variables[f"s{i+1}"])
        #         )

        ### STEP 3 & 4: Define the shared one-way variables and state constraints
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
                            specific_variables[key][(source, target, net_id)] <= shared_variables[(target, source, net_id)]
                            )
                    continue
                var:Variable = self._solver.add_variable(lb = 0, ub =(int(wei) + 1)*len(subnets), name= f'x_{source}_{target}_{net_id}')
                shared_variables[(source, target, net_id)] = var
                avoidable_crossing = False
                e = self._graph.get_edge(source,target)
                # From both direction we only save one, so just if source/target is in modulenames is unavoidable
                if e.crossing and not(e.source.modulename in modulenames) and not(e.target.modulename in modulenames):
                    avoidable_crossing = True
                self._variables.add((source, target, net_id), var, crossing=avoidable_crossing,via=e.via)
                for key in specific_variables:
                    self._solver.add_linear_constraint(
                        specific_variables[key][(source, target, net_id)] <= var
                        )
                # Add the interconnected module varaibles to the shared net variables
                # if source in extra_var:
                #     inter_nodes=[]
                #     for var_dict in extra_var[source]:
                #         var_id = (source, target, net_id)
                #         rev_var_id = (target,source,net_id)
                #         if var_id in var_dict:
                #             inter_nodes.append(var_dict[var_id])
                #         elif rev_var_id in var_dict:
                #             inter_nodes.append(var_dict[rev_var_id])
                #     self._solver.add_linear_constraint(sum(inter_nodes) <= var)
        return [var for key in specific_variables for var in specific_variables[key].values()]
                
    def _add_intermodule_constraints(self, net_id:int, modulename:str, wei:float, subnets_vars:tuple)->dict[NodeId, list[dict[VarId,Variable]]]:

        # Create varaibles for inside the module
        extra_directed_var: dict[NodeId, dict[str,list[Variable]]] = {}
        net_extra_var: dict[NodeId, list[dict[VarId,Variable]]] = {}

        for n in self._graph.get_nodes_by_modulename(modulename):
            for nei in self._graph.adjacent_list[n._id]:
                e = self._graph.get_edge(n._id,nei)
                if not e.crossing:
                    var:Variable = self._solver.add_variable(lb = 0, ub = int(wei) + 1, name= f'extra_{e.source._id}_{e.target._id}_{net_id}')
                    
                    extra_directed_var.setdefault(e.source._id,{}).setdefault('out',[]).append(var)
                    net_extra_var.setdefault(e.source._id,[]).append({(e.source._id,e.target._id,net_id):var})
                    extra_directed_var.setdefault(e.target._id,{}).setdefault('in',[]).append(var)
                    net_extra_var.setdefault(e.target._id,[]).append({(e.source._id,e.target._id,net_id):var})
        # Add the varaibles from the subnet that goes inside
        source_sink_edges = self._graph.get_crossings_by_modulename(modulename)
        for in_edge in source_sink_edges['in']:
            assert in_edge.target.modulename == modulename, "Module name do not match"
            var = subnets_vars[0][(in_edge.source._id,in_edge.target._id,net_id)]
            extra_directed_var.setdefault(in_edge.target._id,{}).setdefault('in',[]).append(var)
        # Add the varaibles from the subnet that goes outside
        for out_edge in source_sink_edges['out']:
            assert out_edge.source.modulename == modulename, "Module name do not match"
            var = subnets_vars[1][(out_edge.source._id,out_edge.target._id,net_id)]
            extra_directed_var.setdefault(out_edge.source._id,{}).setdefault('out',[]).append(var)
        # Add flow conservation constraints for the inside module
        for n in net_extra_var:
            self._solver.add_linear_constraint(sum(extra_directed_var[n]['in']) - sum(extra_directed_var[n]['out']) == 0)
            #self._solver.Add(sum(extra_directed_var[n]['in_sub']) == wei)
            #self._solver.Add(sum(extra_directed_var[n]['out_sub']) == wei)

        return net_extra_var
        
    def _add_sub2pin_net(self, net_id:int, subnet:NamedHyperEdge, subnet_id: str, nodes: list[HananNode3D], edgeids:set[EdgeID])->dict[VarId, Variable]:
        modulenames = subnet.modules
        net_variables: dict[VarId, Variable] ={}
        ### STEP 3: Define the variables
        for e_id in edgeids:
            var:Variable = self._solver.add_variable(lb = 0, ub = int(subnet.weight) + 1, name= f'{subnet_id}_{e_id[0]}_{e_id[1]}_{net_id}')
            net_variables[(e_id[0], e_id[1], net_id)] = var
    
        ### STEP 4: State the constraints
        source_edges = self._graph.get_crossings_by_modulename(modulenames[0])
        sink_edges = self._graph.get_crossings_by_modulename(modulenames[1])
        # Source and Sink are the only nodes that have flow non-zero
        # For source, the outflow is net weigth and the inflow is 0
        self._solver.add_linear_constraint(sum(net_variables[(e.source._id, e.target._id, net_id)] for e in source_edges['out'] if (e.source._id, e.target._id) in edgeids) == subnet.weight)
        self._solver.add_linear_constraint(sum(net_variables[(e.source._id, e.target._id, net_id)] for e in source_edges['in']  if (e.source._id, e.target._id) in edgeids) == 0)
        # For sink, the inflow is the net weight and the outflow is 0
        self._solver.add_linear_constraint(sum(net_variables[(e.source._id, e.target._id, net_id)] for e in sink_edges['in'] if (e.source._id, e.target._id) in edgeids) == subnet.weight)
        self._solver.add_linear_constraint(sum(net_variables[(e.source._id, e.target._id, net_id)] for e in sink_edges['out'] if (e.source._id, e.target._id) in edgeids) == 0)

        # Adding constraints on flow conservation
        for n in nodes:
            # We have to add constraints on all nodes except the source and sink
            if n.modulename in modulenames:
                continue
            in_out_edges: dict[str, list[HananEdge3D]] = self._graph.get_edges_from_node(n)
            in_var = [net_variables[(e.source._id, e.target._id, net_id)] for e in in_out_edges["in"] if (e.source._id, e.target._id) in edgeids]
            out_var = [net_variables[(e.source._id, e.target._id, net_id)] for e in in_out_edges["out"] if (e.source._id, e.target._id) in edgeids]
            self._solver.add_linear_constraint(sum(in_var) - sum(out_var) == 0)
        return net_variables

    def _add_capacity_constraints(self):
        ### STEP 4: Add capacity constraints
        # Add constraints on not exceeding capacity
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
                    vars1 = [var for var in self._variables.get_by_edge_id((source_id, target_id), len(self._nets))]
                    vars2 = [var for var in self._variables.get_by_edge_id((target_id, source_id), len(self._nets))]
                    if len(vars1+vars2) == 0:
                        continue
                    self._solver.add_linear_constraint(
                        sum(vars1 + vars2) <= e.capacity
                    ) # Capacity set here e.capacity

    def solve(self, f_wl:float=0.4, f_mc:float=0.3, f_vu:float=0.3)-> bool:
        assert abs(f_wl + f_mc + f_vu - 1) < 1e-6, "The given factors are not unitary (their sum is not 1)"
        self._m['factors'] = (f_wl,f_mc,f_vu)
        self._add_capacity_constraints()
        ### STEP 5: Define the objective & Compute the normalization factors.
        wire_length = [var * self._graph.get_edge(var_id[0], var_id[1]).length for var_id, var in self._variables.get_all().items()]
        hpwl = 0
        for n in self._nets.values():
            if isinstance(n,HyperEdge):
                centers_x = [m.center.x for m in n.modules if m.center]
                centers_y = [m.center.y for m in n.modules if m.center]
            else:
                centers_x = {c.center.x for m in n.modules for c in self._graph.get_nodes_by_modulename(m)} # Should compute the centroid and not the cell
                centers_y = {c.center.y for m in n.modules for c in self._graph.get_nodes_by_modulename(m)}
            # Manhattan distance
        if centers_x and centers_y:  # Ensure we don't call max/min on empty lists
                hpwl += (max(centers_x) - min(centers_x) + max(centers_y) - min(centers_y)) * n.weight
        self._norms[0]=hpwl

        module_crossings = self._variables.crossings
        self._solver.minimize_linear_objective(sum(module_crossings))    
        status = mathopt.solve(self._solver, mathopt.SolverType.GLOP)
        if self.has_solution(status):
            self._norms[1]=status.objective_value()
        else: # No solution
            return self.has_solution(status)

        via_usage = self._variables.vias
        self._norms[2]=sum(n.weight for n in self._nets.values())

        print(f"Normalization factors>\nwl={self._norms[0]}\nmc={self._norms[1]}\nvu={self._norms[2]}")

        self._solver.minimize_linear_objective(
            f_wl * sum(wire_length)/self._norms[0] +
            f_mc * sum(module_crossings)/self._norms[1] +
            f_vu * sum(via_usage)/self._norms[2] #+ 
            #100 * sum(self.shared) TODO think why
        )
        self._m['norm_fact'] = tuple(self._norms)

        ### STEP 6: Solve the model
        status = mathopt.solve(self._solver, mathopt.SolverType.GLOP)
        self.execution_time = status.solve_time().total_seconds()
        self.is_solved = self.has_solution(status)
        self.result= status
        self._variables.update(status)
        return self.is_solved

    def has_solution(self, status:SolveResult|None=None)->bool:
        if status is None:
            if hasattr(self, 'is_solved'):
                return self.is_solved
            else:
                return None
        msg = status.termination.detail
        if status.termination.reason == mathopt.TerminationReason.OPTIMAL:
            print(f"A probably optimal solution (up to numerical tolerances) has been found.\n{msg}")
            return True
        elif status.termination.reason == mathopt.TerminationReason.FEASIBLE:
            #See SolveResultProto.limit_detail for detailed description of the kind of limit that was reached.
            print(f"The optimizer reached some kind of limit and a primal feasible solution is returned.")
            return True
        elif status.termination.reason == mathopt.TerminationReason.INFEASIBLE:
            print(f"The primal problem has no feasible solutions.\n{msg}")
            return False
        elif status.termination.reason == mathopt.TerminationReason.UNBOUNDED:
            print(f"The primal problem is feasible and arbitrarily good solutions can be found along a primal ray.\n{msg}")
            return False
        elif status.termination.reason == mathopt.TerminationReason.INFEASIBLE_OR_UNBOUNDED:
            # status.solve_stats.to_proto().problem_status
            print(f"The primal problem is either infeasible or unbounded.\n{msg}")
            return False
        elif status.termination.reason == mathopt.TerminationReason.IMPRECISE:
            print("The problem was solved to one of the criteria above (Optimal, Infeasible, Unbounded, or InfeasibleOrUnbounded), ",
                "but one or more tolerances was not met. Users can still query primal/dual solutions/rays and solution stats, but they",
                f" are responsible for dealing with the numerical imprecision.\n{msg}")
            return False
        elif status.termination.reason == mathopt.TerminationReason.NO_SOLUTION_FOUND:
            # See SolveResultProto.limit_detail for detailed description of the kind of limit that was reached.
            print(f"The optimizer reached some kind of limit and it did not find a primal feasible solution.\n{msg}")
            return False
        elif status.termination.reason == mathopt.TerminationReason.NUMERICAL_ERROR:
            print(f"The algorithm stopped because it encountered unrecoverable numerical error. No solution information is present.\n{msg}")
            return False
        elif status.termination.reason == mathopt.TerminationReason.OTHER_ERROR:
            print(f"The algorithm stopped because of an error not covered by one of the statuses defined above. No solution information is present.\n{msg}")
            return False
        else:
            print("Unknown status returned!")
            return False
        
    def has_route_path(self, identifiers: list[dict[EdgeID, float]]) -> bool:
        """Determine if there exists a route path in the given list of EdgesIds"""
        # Parse all identifiers and build a graph
        graph = defaultdict(list)
        for identifier in identifiers:
            fromid = identifier.keys()[0]
            toid = identifier.keys()[1]
            graph[fromid].append(toid)

        # Perform a DFS/BFS to check for a route path
        def dfs(node, visited:set):
            """Helper function to perform DFS."""
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, visited)

        # Find all nodes (keys and values)
        all_nodes = set(graph.keys()).union({to for values in graph.values() for to in values})

        # Check if there is a route path starting from any node
        for start_node in all_nodes:
            visited = set()
            dfs(start_node, visited)
            if visited == all_nodes:
                return True  # All nodes are connected in some path

        return False

    @property
    def solution(self)-> VariableStore:
        if not self.is_solved:
            warnings.warn("No solution exists for the given problem!", UserWarning)
        return self._variables
    
    @property
    def hanan_graph(self) -> HananGraph3D:
        return self._graph
    
    @property
    def metrics(self)->dict[str,int|float]:
        """
        Returns a dict with metrics.

        :return n_nets: an int with number of nets
        :return n_vars: an int with number of variables
        :return n_constraints: an int with number of constraints
        :return elapsed_time_secs: a float with the time taken to solve the model
        :return total_wl: a float with the total wire-length
        :return module_crossings: a float with the number wires that cross an avoidable module
        :return via_usage: a float with the number wires that uses a via
        :return mrd: Maximum Edge Congestion (MRD), the highest congestion value among all edges
        :return n_bifurations: an int with the number of nets that have at least one bifurcation
        :return factors: a tuple (fwl,fmc,fvu) with the used importance factors in the objective function
        :return norm_fact: a tuple (n_wl,n_mc,n_vu) with the used normalization factors
        """
        self._m['n_nets']=len(self._nets)
        self._m['n_vars']=len(list(self._solver.variables()))
        self._m['n_constraints'] = len(list(self._solver.linear_constraints()))
        if hasattr(self,'is_solved'):
            self._m['elapsed_time_secs'] = self.execution_time
            if self.is_solved: ########################################################################
                wire_length = [self.result.variable_values(var) * self._graph.get_edge(var_id[0], var_id[1]).length for var_id, var in self.solution.get_all().items()]
                self._m['total_wl'] = sum(wire_length)
                self._m['module_crossings'] = sum(self.result.variable_values(self.solution.crossings))
                self._m['via_usage'] = sum(self.result.variable_values(self.solution.vias))
                self._m['mrd'] = max([ c/self._graph.get_edge(e[0], e[1]).capacity for e, c in self.solution.congestion.items()])
                self._m['n_bifurations'] = len([1 for netid, route in self.solution.routes.items() if self.has_bifurcation(self._nets[netid], route)])
            else:
                warnings.warn("No solution exists for the given problem!", UserWarning)
            return self._m
        else:
            warnings.warn("Problem is not solved!", UserWarning)
            return self._m  

    def has_bifurcation(self, net: HyperEdge, route: list[dict[EdgeID, float]])->bool:
        """
        Checks whether a route solution has a bifurcation (nodes with degree > 2 or source/sink nodes with degree > 1)
        
        :param route: a list of dict[EdgeId, float] with the float as the number of wires through edge (key).
        :param net: a HyperEdge with the net information

        :return bool:
        """
        #If any node has degree higher than 2, implies a bifurcation
        degrees = compute_node_degrees(route)     
        for n, d in degrees.items():
            if d > 2: 
                return True
        # Check if the source/sink module has degree > 1 (if has two entry/exit implies bifurcation)
        nodes_set = [set([n._id for n in self._graph.get_nodes_by_modulename(m.name)]) for m in net.modules]
        for nodes in nodes_set:
            d= len([n for n in nodes if n in degrees])
            if d > 1:
                return True
        return False

    def save(self, netnames:dict[NetId,str], filepath="./", filename='routes',extension='.txt'):

        with open(f"{filepath}{filename}{extension}", 'w') as f:
            for netid, route in self.solution.routes.items():
                f.write(f"{netnames.get(netid, "n")} {netid} {len(route)}")
                for r in route:
                    f.write(f"({r[0][0]},{r[0][1]},{r[0][2]})-({r[1][0]},{r[1][1]},{r[1][2]})")
                f.write(f"!")

    def set_capacity_adjustments(self, cap_adjust:dict[EdgeID, float|int]):
        self._graph.apply_capacity_adjustments(cap_adjust)