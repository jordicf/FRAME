import odb
import kahypar as kahypar
import sys
import math
from typing import Set, Union
import yaml


KW_AREA = "area"
KW_ASPECT_RATIO = "aspect_ratio"
KW_CENTER = "center"
KW_HARD = "hard"
KW_FIXED = "fixed"
KW_RECTANGLES = "rectangles"
KW_MODULES = "Modules"
KW_NETS = "Nets"
KW_TERMINAL = "terminal"


class Partition:

    def __init__(self, block, net_odb, macros, instance_to_id, design, imbalance: float = 0.3):

        self._macros = macros
        self._instance_to_id = instance_to_id
        self._cluster_to_macro

        hyperedge_indices, hyperedges, edge_weights = obtain_nets_kahypar(net_odb, instance_to_id)
        
        self._k = len(macros) 
        self._num_nodes = len(instance_to_id)
        
        if self._num_nodes == self._k:
            print("This design has no stdcells.")
            print("Each macro is treated as a single cluster.")
            self._cluster_assignment: list[int] = [i for i in range(k)]
            self._cluster_to_macro = clusters_to_hard_modules

        else:
            print("Generating KaHyPar-partition and obtaining clusters.")
            print("Every cluster consists of a macro and multiple stdcells.")
            
            self._num_edges = len(hyperedge_indices) - 1
            self._imbalance = imbalance
            self._node_weights: list[int] = obtain_node_weights(block)
            
            # generate hypergraph
            self._hypergraph = self.createHypergraph()
            # we can choose any of the config files
            self._output_file  = f'{self.design}.part.{self.k}.KaHyPar'

            self._context = self.config_context()
            self.partition()

            # single number per line
            with open(self.output_file) as file:
                cluster_assignment = file.read().splitlines()
            
            self._cluster_to_macro = clusters_to_soft_modules

        self._cluster_assignment = cluster_assignment


    def createHypergraph(self):
        hypergraph = kahypar.Hypergraph(self.num_nodes, self.num_edges, self.hyperedge_indices, self.hyperedges, self.k, self.edge_weights, self.node_weights)
        self.fix_macros()


    def num_nodes(self):
        return self._num_nodes
    

    def num_edges(self):
        return self._num_edges


    def hyperedge_indices(self):
        return self._hyperedge_indices

    
    def hyperedges(self):
        return self._hyperedges


    def k(self):
        return self._k


    def design(self):
        return self._design

    
    def edge_weights(self):
        return self._edge_weights


    def node_weights(self):
        return self._node_weights

    
    def macros(self):
        return self._macros

    
    def instance_to_id(self):
        return self._instance_to_id


    def output_file(self):
        return self._output_file


    def imbalance(self):
        return self._imbalance


    def fix_macros(self):
        # fixar els macros
        for i in range(self.k):
            node_id = self.instance_to_id[self.macros[i]]
            self._hypergraph.fixNodeToBlock(node_id, i)

    
    def config_context(self, p_file: str = "../kahypar/config/km1_kKaHyPar_sea20.ini"):
        # set context
        context = kahypar.Context()
        context.loadINIconfiguration(p_file)
        context.setK(self.k)
        context.setEpsilon(self.imbalance)
        context.setPartitionFileName(self.output_file)
        context.writePartitionFile(True)
        return context
                    

    def partition(self):
        # generate partition
        kahypar.partition(self._hypergraph, self._context)


class ODBtoFRAME:
    """
    Class to translate from ODB format to FRAME FPEF.
    """

    _block
    _net_odb

    def __init__(self, design: str, design_directory: str = "."):
        """
        Constructor of a translator.
        :parap design: name of the design of the .odb file.
        :param design_directory: directory in which to find the .odb file.
        """

        odb_file = design_directory + design + ".odb"
        
        print("Creating Database.")
        db = odb.dbDatabase.create()
        odb.read_db(db, odb_file)

        self._block = db.getChip().getBlock()
        self._net_odb = self._block.getNets()

        print("Creating Bundled IOs.")
        io_cluster_map = create_bundled_IOs(db)

        print("Obtaining die, modules and net for clusterization.")
        self._die_dict = self.compute_die()

        macros, instance_to_id = self.obtain_instances() # són els nodes
        
        self._partition = Partition(self.block, self.net_odb, macros, instance_to_id, design)

        self._modules: dict[dict] = self.clusters_to_macros(block, instance_to_id, partition) # per frame



    def design(self):
        return self._design
    

    def block(self):
        return self._block


    def net_odb(self):
        return self._net_odb