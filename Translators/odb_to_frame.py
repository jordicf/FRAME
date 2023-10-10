
import odb
import argparse
import kahypar as kahypar
import sys
import math
from typing import Set, Union, Any
import yaml
import os
# from frame.utils.keywords import KW_MODULES, KW_NETS, KW_AREA, KW_CENTER, KW_TERMINAL, KW_MIN_SHAPE, KW_ASPECT_RATIO


KW_AREA = "area"
KW_ASPECT_RATIO = "aspect_ratio"
KW_CENTER = "center"
KW_HARD = "hard"
KW_FIXED = "fixed"
KW_RECTANGLES = "rectangles"
KW_MODULES = "Modules"
KW_NETS = "Nets"
KW_TERMINAL = "terminal"



def map_IO_pads(block):

    """
    TODO: Understand this function. Used in openroad when creating bundled IOs.
    """
    is_pad_design = False
    for inst in block.getInsts():
        if inst.getMaster().isPad(): # pad is similar to a terminal 
            is_pad_design = True
            break
    
    if not is_pad_design:
        return {}
    
    io_pad_map = {}
    for net in block.getNets():
        if len(net.getBTerms()) > 0: 
            for bterm in net.getBTerms():
                for iterm in net.getITerms():
                    instance = iterm.getInst()
                    io_pad_map[bterm] = instance
    return io_pad_map


def create_bundled_IOs(db, dbu):
    """
    Given a Database (in odb format) create the terminal (BTerm) clusters so as to
    simplify the design. Returns a dictionary (cluster_map) with the clusters 
    (x,y,width,height,[list of terminals]).
     
    Function extracted from Openroad. Originally not in Python.
    """

    chip = db.getChip()
    block = chip.getBlock()
    die_box = block.getDieArea()

    # dbu = db.getTech().getDbUnitsPerMicron()
    floorplan_lx = die_box.xMin()
    floorplan_ly = die_box.yMin()
    floorplan_ux = die_box.xMax()
    floorplan_uy = die_box.yMax()
    num_bundled_IOs = 3 # default value is 3

    # Get the floorplan information and get the range of bundled IO regions
    die_box = block.getCoreArea()
    core_lx = die_box.xMin()
    core_ly = die_box.yMin()
    core_ux = die_box.xMax()
    core_uy = die_box.yMax()
    x_base = (floorplan_ux - floorplan_lx) / num_bundled_IOs
    y_base = (floorplan_uy - floorplan_ly) / num_bundled_IOs

    cluster_id = 0  
    cluster_id_base = cluster_id

    io_pad_map = map_IO_pads(block) # empty dictionary if there are no pads 
    
    # we model each bundled IO as a cluster under the root node
    # Map all the BTerms / Pads to Bundled IOs (cluster)
    prefix_vec = ["L", "T", "R", "B"]
    cluster_map = {}

    # create the (4 x num_bundled_IOs) clusters
    for i in range(4): # four boundaries (Left, Top, Right and Bottom in order)
        for j in range(num_bundled_IOs):
            cluster_name = "IO_cluster_" + prefix_vec[i] + str(j)

            x, y, width, height = 0.0, 0.0, 0, 0
            if i == 0: # Left boundary
                x = floorplan_lx
                y = floorplan_ly + y_base * j
                height = y_base
            elif i == 1:  # Top boundary
                x = floorplan_lx + x_base * j
                y = floorplan_uy
                width = x_base
            elif i == 2 :  # Right boundary
                x = floorplan_ux
                y = floorplan_uy - y_base * (j + 1)
                height = y_base
            else:   # Bottom boundary
                x = floorplan_ux - x_base * (j + 1)
                y = floorplan_ly
                width = x_base
            # dbuToMicron(x, y, width, height)
            cluster_map[cluster_id] = {"name": cluster_name, "x": x/dbu,  "y": y/dbu, "width": width/dbu,"height": height/dbu, "terminals" : []}
            #  set the cluster to an IO cluster
            cluster_id += 1


    # Map all the BTerms to bundled IOs
    for bterm in block.getBTerms():
        lx, ly = sys.maxsize, sys.maxsize
        ux, uy = 0, 0
        # If the design has IO pads, these block terms
        # will not have block pins.
        # Otherwise, the design will have IO pins.
        for pin in bterm.getBPins():
            for box in pin.getBoxes():
                lx = min(lx, box.xMin())
                ly = min(ly, box.yMin())
                ux = max(ux, box.xMax())
                uy = max(uy, box.yMax())

        # remove power pins
        if bterm.getSigType() == "Supply":
            continue
        
        # If the term has a connected pad, get the bbox from the pad inst
        if bterm in io_pad_map: # aquest if no n'estic segura
            lx = io_pad_map[bterm].getBBox().xMin()
            ly = io_pad_map[bterm].getBBox().yMin()
            ux = io_pad_map[bterm].getBBox().xMax()
            uy = io_pad_map[bterm].getBBox().yMax()
            if lx <= core_lx :
                lx = floorplan_lx
            if ly <= core_ly :
                ly = floorplan_ly
            if ux >= core_ux :
                ux = floorplan_ux
            if uy >= core_uy :
                uy = floorplan_uy

        # calculate cluster id based on the location of IO Pins / Pads
        cluster_id = -1
        if lx <= floorplan_lx:
        # The IO is on the left boundary
            cluster_id = cluster_id_base + math.floor(((ly + uy) / 2.0 - floorplan_ly) / y_base)
        elif uy >= floorplan_uy:
            # The IO is on the top boundary
            cluster_id = cluster_id_base + num_bundled_IOs + math.floor(((lx + ux) / 2.0 - floorplan_lx) / x_base)
        elif ux >= floorplan_ux: 
            # The IO is on the right boundary
            cluster_id = cluster_id_base + num_bundled_IOs * 2 + math.floor((floorplan_uy - (ly + uy) / 2.0) / y_base)
        elif ly <= floorplan_ly:
            # The IO is on the bottom boundary
            cluster_id = cluster_id_base + num_bundled_IOs * 3 + math.floor((floorplan_ux - (lx + ux) / 2.0) / x_base)
        if cluster_id == -1:
            print(f"Floorplan has not been initialized? Pin location error for {bterm.getName()}")
        else:
            cluster_map[cluster_id]["terminals"].append(bterm.getName())
        

    cluster_map_ret = cluster_map.copy()
    # delete the IO clusters that do not have any pins assigned to them
    for cluster_id in cluster_map:
        if len(cluster_map[cluster_id]["terminals"]) == 0:
            cluster_map_ret.pop(cluster_id)

    return cluster_map_ret


def compute_die(block, dbUnitsPerMicron):
    """
    Returns coordinates from the four tips of the die.
    (xmin, ymax) ----------------- (xmax, ymax)
        |                                |
        |                                |
        |                                |
        |                                |
        |                                |
        |                                |
    (xmin, ymin)------------------- (xmax, ymin)
    """


    bbox = block.getCoreArea() # bounding box, dimensions del die
    die_dict = {"width": (bbox.xMax() - bbox.xMin())/dbUnitsPerMicron, "height": (bbox.yMax() - bbox.yMin())/dbUnitsPerMicron} 
    return die_dict


def obtain_node_weights(block):
    """
    Given a DataBase (odb) block, it returns the node's weights.
    Only stdcells (CORE) and macros (BLOCK).
    """

    insts = block.getInsts()
    node_weights: list[float] = []
    for inst in insts:
        if inst.isCore() or inst.isBlock():
            bbox = inst.getBBox()
            area = (bbox.xMax() - bbox.xMin())*(bbox.yMax() - bbox.yMin())
            node_weights.append(int(area))

    units = block.getDefUnits()
    node_weights = [int(i/units) for i in node_weights]
    return node_weights


def obtain_instances(block):
    """
    Given a database's (odb) block, it returns:
        1. List of macros' names
        2. Dictionary that maps each instance name to its node_id
    """

    insts = block.getInsts() # instances are stdcells (CORE) and macros (BLOCK)
    macros: list[str] = [] 
    instance_to_id: dict[str, int] = {} # dict. key: instance name value: node id

    node_id = 0
    for inst in insts:
        if inst.isCore():
            instance_to_id[inst.getName()] = node_id
            node_id = node_id + 1
        elif inst.isBlock():          
            macros.append(inst.getName())
            instance_to_id[inst.getName()] = node_id
            node_id = node_id + 1

    return macros, instance_to_id


def obtain_nets_kahypar(net_odb, instance_to_id):
    """
    Obtains nets from the database.
    
    net_odb: Net of a database in odb format
    instance_to_id: dictionary that maps instance to its node id

    :returns: lists that are necessary in order to generate the KaHyPar graph:
    hyperedge_indices, hyperedges, weights.
    Does not take into account border terminals (BTerms).
    
    -------------------------------
    Annotation:
    
        BTerms versus Iterms

        A block-terminal is the element used to represent connections in/out of
        a block. 

        An instance-terminal is the element used to represent connections in/out of
        an instance. Found in elements inside the die like macros or stdcells.
        
    """

    hyperedge_indices, hyperedges, weights = [], [], []

    it = 0
    for edge_odb in net_odb:
        iterms = edge_odb.getITerms() # instance terminals
        if len(iterms) > 1:
            hyperedge_indices.append(it)
            sub_hyperedge = set() # use set because some macros may appear more than once
            for iterm in iterms:
                instance = iterm.getInst()
                sub_hyperedge.add(instance_to_id[instance.getName()])
            hyperedges = hyperedges + list(sub_hyperedge)
            it = it + len(sub_hyperedge)
            weights.append(edge_odb.getWeight())

    hyperedge_indices.append(it)    
    return hyperedge_indices, hyperedges, weights


def clusters_to_hard_modules(block, instance_to_id, cluster_assignment, dbUnitsPerMicron):
    """
    If there are only macros and no stdcells, each macro represents
    a cluster in itself. In FRAME those would be represented by hard modules.
    
    block: Block from the database
    instance_to_id: dictionary that associates instance name to its node id
    cluster_assignment: list with the cluster assignment

    :returns: dictionary of modules (doesn't include IO clusters)
    """
    
    modules = {}
    for instance_name in instance_to_id:
        id = instance_to_id[instance_name]
        cluster = cluster_assignment[id]
        instance = block.findInst(instance_name)
        name_module = "M_" + str(cluster)
        
        bbox = instance.getBBox()
        x = (bbox.xMin() + bbox.xMax())/2/dbUnitsPerMicron
        y = (bbox.yMin() + bbox.yMax())/2/dbUnitsPerMicron
        width = (bbox.xMax() - bbox.xMin())/dbUnitsPerMicron
        height = (bbox.yMax() - bbox.yMin())/dbUnitsPerMicron

        modules[name_module] = {KW_HARD: True, KW_RECTANGLES: [x, y,width, height]}

    return modules


def clusters_to_soft_modules(block, instance_to_id, cluster_assignment, dbUnitsPerMicron):
    """
    If there are stdcells, every module/cluster is composed by 1 macro and many
    various stdcells. In FRAME those would be represented by soft modules.
    
    block: Block from the database
    instance_to_id: dictionary that associates instance name to its node id
    cluster_assignment: list with the cluster assignment

    :returns: dictionary of modules (doesn't include IO clusters)
    """

    modules = {}
    cluster_to_num_items = {}

    for instance_name in instance_to_id:
        id = instance_to_id[instance_name]
        cluster = int(cluster_assignment[id])
        instance = block.findInst(instance_name)

        name_module = "M_" + str(cluster)
        bbox = instance.getBBox()
        area = (bbox.xMax() - bbox.xMin())*(bbox.yMax() - bbox.yMin())/(dbUnitsPerMicron**2)
        center = [(bbox.xMin() + bbox.xMax())/2/dbUnitsPerMicron, (bbox.yMin() + bbox.yMax())/2/dbUnitsPerMicron]
         
        if name_module not in modules: # inicialize
            modules[name_module] = {KW_AREA : area, KW_CENTER : center}
            cluster_to_num_items[cluster] = 1
        else:
            modules[name_module][KW_AREA] += area
            modules[name_module][KW_CENTER][0] += center[0]
            modules[name_module][KW_CENTER][1] += center[1]
            cluster_to_num_items[cluster] += 1

        if instance.isBlock():
            min_shape = [(bbox.xMax() - bbox.xMin())/dbUnitsPerMicron, (bbox.yMax() - bbox.yMin())/dbUnitsPerMicron]
            modules[name_module]["min_shape"] = min_shape        

    for name_module in modules:
        num_items = cluster_to_num_items[int(name_module.split("_")[-1])]
        modules[name_module][KW_CENTER][0] /= num_items
        modules[name_module][KW_CENTER][1] /= num_items

        min_x, min_y = modules[name_module]["min_shape"]
        modules[name_module][KW_ASPECT_RATIO] = modules[name_module][KW_AREA] / (min(min_x, min_y)**2)
        modules[name_module].pop("min_shape")

    return modules


def get_cluster_name(inst, element_to_cluster):
    """
    Returns name of a cluster instance.
    inst: isntance which cluster we want to obtain
    element_to_cluster: dictionary that associates databas instances to the
            name of the cluster they belong to.
    """
    return element_to_cluster[inst.getName()]


def calculate_connections(net_odb, element_to_cluster: dict[str, str]):
    """
    Returns the resulting netlist after clustering.
    net_odb: original netlist
    element_to_cluster: dictionary that associates database instances to
                    the name of the cluster they belong to.
    """

    virtual_weight = 10.0 
    netlist: dict[list, float] = {}

    for net in net_odb:
        if net.getSigType() == "SUPPLY":
            continue
        
        driver_id = -1 # cluster id of the driver instance (sortida)
        loads_id = {} # cluster id of the sink instances (arribades)
        pad_flag = False
        for iterm in net.getITerms():
            inst = iterm.getInst()
            master = inst.getMaster()
        # check if the instance is a Pad, Cover or empty block (such as marker)
        # We ignore nets connecting Pads, Covers, or markers
            if master.isPad() or master.isCover():
                pad_flag = True
                break

            cluster_name = get_cluster_name(inst, element_to_cluster)
            if iterm.getIoType() == 'OUTPUT': #mirar això
                driver_id = cluster_name
            elif cluster_name not in loads_id:
                loads_id[cluster_name] = 1
            else:
                loads_id[cluster_name] += 1
        if pad_flag:
            continue 
        
        io_flag = False

        # check the connected IO pins
        for bterm in net.getBTerms():
            io_flag = True # hi ha bterms
            cluster_name = get_cluster_name(bterm, element_to_cluster)
            if bterm.getIoType() == "INPUT" :
                driver_id = cluster_name
            elif cluster_name not in loads_id:
                loads_id[cluster_name] = 1
            else:
                loads_id[cluster_name] += 1 

        # add the net to the netlist
        if driver_id != -1 and len(loads_id) > 0:
            weight = net.getWeight() if (io_flag == True) else virtual_weight
            for cluster_name in loads_id:
                if cluster_name < driver_id:
                    if cluster_name + " + " + driver_id in netlist:
                        netlist[cluster_name + " + " + driver_id] += weight*loads_id[cluster_name]
                    else:
                        netlist[cluster_name + " + " + driver_id] = weight*loads_id[cluster_name]
                elif cluster_name > driver_id:
                    if driver_id + " + " + cluster_name in netlist:
                        netlist[driver_id + " + " + cluster_name] += weight*loads_id[cluster_name]
                    else:
                        netlist[driver_id + " + " + cluster_name] = weight*loads_id[cluster_name]                    

    netlist_frame = []
    for net in netlist:
        netlist_frame.append(net.split(" + ") + [netlist[net]])

    return netlist_frame
  

def write_output(output_dir: str, design: str, netlist, die_dict: dict[str, float], element_to_cluster: dict[str, str]) -> None:
    """
    Writes output to YAML files.

    output_dir: directory to write the output yaml file.
    design: name of the design
    netlist: dictionary that contains info from modules and from net
    die_dict: dictionary that contains info of die (width and height)
    element_to_cluster: dictionary that maps each instance (stdcell, macro, IO pin, etc) to its cluster.
    """

    netlist_path = output_dir + "/netlist_" + design + ".yaml"
    with open(netlist_path, "w") as f_netlist:
        documents = yaml.dump(netlist, f_netlist)

    die_path = output_dir + "/die_" + design + ".yaml"
    with open(die_path, "w") as f_die:
        documents = yaml.dump(die_dict, f_die)
        
    element_to_cluster_path = output_dir + "/element_to_cluster" + ".yaml"
    with open(element_to_cluster_path, "w") as f_cluster_map:
        documents = yaml.dump(element_to_cluster, f_cluster_map)


def parse_options(args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the translator.
    """
    
    parser = argparse.ArgumentParser(prog="Format Translator",
                                     description="Obtains the die and netlist structure in FPEF+DIEF format from an .odb file generated through openroad.")
    parser.add_argument("--imbalance", help="Imbalance for Hypergraph partitioning.", default=0.03)
    parser.add_argument("-o", "--optimization", choices=['cut', 'km1'], help="cut: connectivity-1\n cut: cut net.", default="cut")
    parser.add_argument("--design", "-d", required=True, help="Path of the design (odb file).")
    parser.add_argument("--verbose", "-v", action= "store_true", help="Print additional information.")
    parser.add_argument("--kahypar_path", help="Path Kahypar (for partition files).", default = ".")
    parser.add_argument("--output_dir", help="Output directory.")
    return vars(parser.parse_args(args))


def main(args: list[str] | None = None):
    """
    Format translator that obtains the die and netlist structure in FPEF+DIEF format
    from an .odb file generated through openroad.
    It is meant to create a netlist representing a floorplan right before macro placement.
    
    ----------------------------------------------------------------------------------------
    
    Structure:
        1. Read .odb database
        2. Create bundled IOs / terminals: group terminals
        3. Obtain die measurements (height and width)
        
        TODO: Resize so that the numbers are not too big - dbu = db.getTech().getDbUnitsPerMicron()
            gets db units.
            
        4. Obtain instances (stdcells, macros)
        5. Obtain nets. This information is useful for the next step, which consists in 
        partitioning the hypergraph and grouping stdcells and macros.
        6. Hypergraph partitioning - using KaHyPar. Each cluster consists of 1 macro 
        and multiple stdcells.
        7. Add resulting soft modules to FRAME netlist. Add bundled IOs to FRAME netlist.
        8. Ccompute connections between elements of the resulting netlist. 
        The edge weight denotes the number of connections between two given modules.
        9. Write output to yaml files. 3 output files: die file, netlist file, cluster file 

    ----------------------------------------------------------------------------------------
    """
    
    options = parse_options(args)

    design_path: str = options["design"]
    design = design_path.split("/")[-1][:-4]
    verbose: bool = options["verbose"]
    output_dir = options["output_dir"]
    
    if not os.path.exists(output_dir):
        print("[INFO]  Error! ",  output_dir, " does not exist!\n")
        exit()
        
    if not os.path.exists(design_path):
        print("[INFO]  Error! ",  design_path, " does not exist!\n")
        exit()
        
    if verbose:
        print("Creating Database.")

    db = odb.dbDatabase.create()
    odb.read_db(db, design_path)
    chip = db.getChip()
    block = chip.getBlock()
    net_odb = block.getNets()
    
    dbUnitsPerMicron = db.getTech().getDbUnitsPerMicron()
    
    if verbose:
        print("Creating Bundled IOs.")
    
    io_cluster_map = create_bundled_IOs(db, dbUnitsPerMicron)

    if verbose:
        print("Obtaining die, modules and net for clusterization.")
        
    die_dict = compute_die(block, dbUnitsPerMicron)
    macros, instance_to_id = obtain_instances(block)
    hyperedge_indices, hyperedges, edge_weights = obtain_nets_kahypar(net_odb, instance_to_id)
    
    k = len(macros) 
    num_nodes = len(instance_to_id)
    
    if verbose:
        print("NUM NODES:", num_nodes)
        print("NUM EDGES:", len(hyperedge_indices))
        print("NUM CLUSTERS:", k)
    
    if k == 0:
        if verbose:
            print("THIS DESIGN HAS NO MACROS!!! EXITING.")
        
        exit()
        
    if num_nodes == k:
        if verbose:
            print("This design has no stdcells.")
            print("Each macro is treated as a single cluster.")
        
        cluster_assignment: list[int] = [i for i in range(k)]
        clusters_to_macros = clusters_to_hard_modules
        
        if verbose:
            print("Translating from clusters to hard macros.")

    else:
        if verbose:
            print("Generating KaHyPar-partition and obtaining clusters.")
            print("Every cluster consists of a macro and multiple stdcells.")
        
        num_edges = len(hyperedge_indices) - 1
        imbalance = options["imbalance"]
        node_weights: list[int] = obtain_node_weights(block)
        
        # generate hypergraph
        hypergraph = kahypar.Hypergraph(num_nodes, num_edges, hyperedge_indices, hyperedges, k, edge_weights, node_weights)

        # fix macros
        for i in range(k):
            node_id = instance_to_id[macros[i]]
            hypergraph.fixNodeToBlock(node_id, i)
        
        if options["optimization"] == "cut":
            p_file: str = options["kahypar_path"] + "/config/km1_kKaHyPar_sea20.ini"
        else:
            p_file: str = options["kahypar_path"] + "/config/cut_kKaHyPar_sea20.ini"
            
        output_file  = f'{design}.part.{k}.KaHyPar'

        # set KaHyPar context
        context = kahypar.Context()
        context.loadINIconfiguration(p_file)
        context.setK(k)
        context.setEpsilon(imbalance)
        context.setPartitionFileName(output_file)
        context.writePartitionFile(True)

        # Generate partition
        kahypar.partition(hypergraph, context)
        
        # single number per line
        with open(output_file) as file:
            cluster_assignment = file.read().splitlines()
        
        clusters_to_macros = clusters_to_soft_modules
        
        if verbose:
            print("Translating from clusters to soft macros.")
    
    # modules' dictionary
    modules: dict[dict] = clusters_to_macros(block, instance_to_id, cluster_assignment, dbUnitsPerMicron) 
    
    # finally, we add the IO clusters
    for cluster_id in io_cluster_map:
        io_cluster = io_cluster_map[cluster_id]
        modules[io_cluster["name"]] = {KW_FIXED: True, KW_TERMINAL: True, KW_CENTER: [io_cluster["x"], io_cluster["y"]]}

    # create element (name) to cluster (name) mapping
    element_to_cluster = {}
    for instance_name in instance_to_id:
        id = instance_to_id[instance_name]
        element_to_cluster[instance_name] = "M_" + cluster_assignment[id]
    
    for io_cluster in io_cluster_map:
        terminals = io_cluster_map[io_cluster]["terminals"]
        for terminal in terminals:
            element_to_cluster[terminal] = io_cluster_map[io_cluster]["name"]

    if verbose:
        print("Calculating connections between clusters.")
    
    netlist = calculate_connections(net_odb, element_to_cluster)

    frame_netlist = {KW_MODULES: modules, KW_NETS: netlist}
    
    if verbose: 
        print("Writing output yaml netlist.")
    
    write_output(output_dir, design, frame_netlist, die_dict, element_to_cluster)


if __name__ == "__main__":
    main()
