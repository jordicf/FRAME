#############################################################
# Convert from open-source protocol buffer format to Floorplan
# Exchange Format.
# Read initial placement and convert to Die Exchange Format.
#############################################################

import os
import typing
import yaml


class ProBufFormat2FPEF:
    """
    Class to transform from Protocol Buffer Format to FPEF format.
    """

    design: str                 # chip design name
    netlist_file: str           # path to netlist file (ProBufFormat)
    plc_file: str               # path to initial placement file (ProBufFormat)
    output_dir: str             # path to output directory (FPEF)
    modules: dict[dict]         # dictionary of modules (key: name)
    nets: list[list]            # list of nets

    def __init__(self, file_dir: str, design: str, output_dir: str = "."):
        """
        Constructs the ProBufFormat2FPEF object.
        """

        self.netlist_file = file_dir + "/netlist.pb.txt"
        self.plc_file = file_dir + "/initial.plc"
        self.design = design
        self.output_dir = output_dir
        self.modules = {}    
        self.nets = []

        # die information
        self.die_dict = {}

        # functions
        self.check_files()
        self.read_plc_file()
        self.read_netlist_file()
        self.write_output()


    def check_files(self) -> None:
        """
        Checks whether paths to netlist_file or plc_file exist. 
        Checks whether output_dir exists.
        """

        if not os.path.exists(self.netlist_file):
            print("[INFO]  Error! ",  self.netlist_file, " does not exist!\n")
            exit()

        if not os.path.exists(self.plc_file):
            print("[INFO]  Error! ",  self.plc_file, " does not exist!\n")
            exit()

        if not os.path.exists(self.output_dir):
            print("[INFO]  Error! ",  self.output_dir, " does not exist!\n")
            exit()


    def read_netlist_file(self) -> None:
        """
        Reads Netlist file.
        """

        with open(self.netlist_file) as f:
            content = f.read().splitlines()
        f.close()
        
        i = 0
        while i < len(content):

            if content[i].startswith("node"): # read_node

                module_read = {"width" : "0.0", "height" : "0.0"}
                
                i += 1
                name = content[i].split()[-1].strip('"')
                i += 1

                # read net
                net = []
                while content[i].startswith("  input"):
                    in_node = content[i].split()[-1].strip('"').split("/")
                    
                    if len(in_node) > 1:
                        in_node.pop()
                        in_node = '/'.join(in_node)
                    else:
                        in_node = in_node[0]

                    net.append(in_node) # macro / pin
                    i += 1

                # read attributes
                while content[i].startswith("  attr"):
                    i += 1
                    key = content[i].split()[-1].strip('"')
                    i += 2
                    module_read[key] = content[i].split()[-1].strip('"')
                    i += 3
                
                
                # for now, we are removing macro_pins
                if module_read["type"].lower() == "macro_pin": 
                    if module_read.get("weight") is not None:
                        for el in net:
                            self.nets.append([el, module_read["macro_name"], float(module_read["weight"])])
                    else:
                        for el in net:
                            self.nets.append([el, module_read["macro_name"]])
                
                else: 
                    if module_read.get("weight") != None:
                        for el in net:
                            self.nets.append([el, name, float(module_read["weight"])])
                    else:
                        for el in net:
                            self.nets.append([el, name])

                    # name = name + module_read["type"][0].lower()  # name also contains type - for inverse translation
                    
                    module_write = {}
                    # width and height cannot be zero
                    # hard and fixed do not have attribute area
                                        
                    
                    if module_read["type"].upper() == "PORT":
                        module_write["fixed"] = True
                        module_write["terminal"] = True
                        module_write["center"] = [float(module_read["x"]), float(module_read["y"])]
                    else:
                        module_write["hard"] = True
                        module_write["rectangles"] = [[float(module_read["x"]), float(module_read["y"]), float(module_read["width"]), float(module_read["height"])]]
                                              
                    self.modules[name] = module_write
            
            i += 1

    def read_plc_file(self) -> None:
        """
        Reads initial placement file and extracts die information: 
        width, height and blockages.
        """

        with open(self.plc_file) as f:
            content = f.read().splitlines()
        f.close()

        i = 0
        while not content[i].startswith("# Width :"): i += 1

        die_dim = content[i].split()
        self.die_dict["width"] = float(die_dim[3])
        self.die_dict["height"] = float(die_dim[6])

        # blockages
        self.die_dict["rectangles"] = []
        while i < len(content) and not content[i].startswith("# node_index"):
            if content[i].startswith("# Blockage :"):
                b = content[i].split()
                self.die_dict.append([float(b[i]) for i in range(3,8)] + ['#'])
            i += 1
    

    def write_output(self) -> None:
        """
        Writes output to YAML files.
        """

        netlist_path = self.output_dir + "/netlist_" + self.design + ".yaml"
        with open(netlist_path, "w") as f_netlist:
            netlist = {"Modules" : self.modules, "Nets" : self.nets}
            documents = yaml.dump(netlist, f_netlist)

        die_path = self.output_dir + "/die_" + self.design + ".yaml"
        if len(self.die_dict["rectangles"]) == 0:
            self.die_dict.pop("rectangles")

        with open(die_path, 'w') as f_die:
            documents = yaml.dump(self.die_dict, f_die)


# in order to compute the cost of a certain placement in FPEF format
# given that the original test was in PBF, we shall overwrite the location
# of the elements from the original file using the final placement file.
class FPEF2ProBufFormat:
    """
    Class to transform from Protocol Buffer Format to FPEF format.
    """

    pbf_path: str
    fpef_path: str
    output_dir: str

    def __init__(self, pbf_path: str, fpef_path: str, output_dir: str = "."):
        """
        Constructs the ProBufFormat2FPEF object.

        Args:
            pbf_path: Path to the original netlist file in Protocol Buffer Format.
            fpef_path: Path to the FPEF format file containing updated coordinates.
            output_dir (optional): Output directory for the final netlist file. Defaults to current directory.
        """

        self.pbf_path = pbf_path
        self.fpef_path = fpef_path
        self.output_dir = output_dir

        # functions
        self.check_files()
        self.obtain_final_netlist_file()


    def check_files(self) -> None:
        """
        Checks whether paths to FPEF netlist or PBF netlist exist. 
        Checks whether output_dir exists.
        If any of the paths do not exist, it raises an error and exits the program.        
        """

        if not os.path.exists(self.fpef_path):
            print("[INFO]  Error! ",  self.fpef_path, " does not exist!\n")
            exit()

        if not os.path.exists(self.pbf_path):
            print("[INFO]  Error! ",  self.pbf_path, " does not exist!\n")
            exit()

        if not os.path.exists(self.output_dir):
            print("[INFO]  Error! ",  self.output_dir, " does not exist!\n")
            exit()


    def obtain_final_netlist_file(self):
        """
        Read original netlist file in ProBufFormat and replace new coordinates
        from FPEF format file. The new netlist is saved in the output directory.
        """

        # Read the original netlist file in PBF format
        with open(self.pbf_path, "r") as f:
            content = f.read().splitlines()

        # Load the FPEF format file
        with open(self.fpef_path) as stream:
            doc = yaml.safe_load(stream)

        i = 0
        while i < len(content):
            if content[i].startswith("  name"):
                name = content[i].split()[-1].strip('"')

                while not content[i].endswith('"type"'):
                    i += 1
                
                i += 2
                nodeType = content[i].split()[-1].strip('"')

                update_location = True # True by default

                if nodeType.upper() == "MACRO_PIN":
                    # find the name of the macro
                    name = content[i - 6].split()[-1].strip('"')
                    while not content[i].endswith('"x"') and not content[i] == "}":
                        i += 1
                    
                    if content[i] == "}": 
                        # no need to update x, y coordinates macro pin
                        update_location = False


                if update_location:    
                    # Get module information from the FPEF format file                
                    module_info = doc["Modules"][name]

                    while not content[i].endswith('"x"'):
                        i += 1
                    
                    # update the location
                    i += 2
                    
                    content[i] = f"      f: {module_info['center'][0]}"
                    i += 6
                    content[i] = f"      f: {module_info['center'][1]}"
            
            i += 1

        # Write the final netlist to the output directory
        output_path = os.path.join(self.output_dir, "final_netlist.pb.txt")
        with open(output_path, "w") as f:
            f.write("\n".join(content))
