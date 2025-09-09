
import distinctipy # pyright: ignore[reportMissingTypeStubs]
from typing import Iterator
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy, QLineEdit,
    QGridLayout, QGroupBox, QCheckBox, QButtonGroup,QMessageBox, QComboBox, QGraphicsItem,
    QTabWidget, QLayout, QGraphicsRectItem, QRadioButton, QAbstractButton, QGraphicsScene
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QBrush, QColor, QResizeEvent
from graphical_view import GraphicalView
from tools.floor_designer.items import Module, FlyLine, RectObj

from frame.netlist.netlist import Netlist 
from frame.geometry.geometry import Rectangle, Point, Shape
from frame.die.die import Die

MIN_NET_PEN = 1
MAX_NET_PEN = 4

class FloorplanDesigner(QWidget):
    """A widget for designing a chip floorplan by placing and managing rectangular modules.
    The widget provides an interactive graphical view where users can select, move and edit
    modules that are loaded from a netlist."""

    _graphical_view: GraphicalView
    _modules: dict[str,Module] # key: name, value: module
    _selected_mod: Module|None = None
    _info_layout: QVBoxLayout # Layout for displaying module information
    _netlist: Netlist
    _fly_lines: set[FlyLine]
    _fly_lines_button_group: QButtonGroup # Manage the visibility of the fly lines
    _blockages: set[QGraphicsRectItem]

    def __init__(self):
        super().__init__()

        # die = Die("tools\\floor_designer\\Examples\\die_dim.yml")
        # die = Die("tools\\floor_designer\\Examples\\die_for_blck.yml")
        die = Die("tools\\floor_designer\\Examples\\DIEF_4.yaml")

        self._graphical_view = GraphicalView(int(die.width),int(die.height))
        self._graphical_view.scene().selectionChanged.connect(self._selected_module_changed)
        self._graphical_view.scene().setParent(self)
        self._modules = dict[str,Module]()
        self._blockages = set[QGraphicsRectItem]()

        # self._netlist = Netlist("tests\\frame\\netlist\\netlist_rect.yml")
        self._netlist = Netlist("tools\\floor_designer\\Examples\\FPEF_4.yaml")
        # self._netlist = Netlist("tools\\floor_designer\\Examples\\legalize.yml")
        # self._netlist = Netlist("tools\\floor_designer\\Examples\\netlist_for_blck.yml")

        self.load_blockages(die)
        self.load_modules()
        self.load_fly_lines()

        v_layout = QVBoxLayout()
        v_layout.addWidget(self._create_info_box())
        v_layout.addWidget(self._create_fly_lines_box())
        v_layout.addWidget(self._create_visual_details_box())

        h_layout = QHBoxLayout()
        h_layout.addWidget(self._graphical_view)
        h_layout.addSpacing(10)
        h_layout.addLayout(v_layout)
        self._update_info_box()

        save_button = QPushButton("Save")
        save_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        save_button.clicked.connect(self._ask_how_to_save)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(save_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 5, 15, 15)
        layout.setSpacing(0)
        layout.addLayout(h_layout)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _selected_module_changed(self) -> None:
        """Manages selection changes in the scene.
        - If a different module is selected, the previously selected one in regrouped and the
        new module is moved to the front. 
        - If a rectangle inside a module is selected, the parent module remains assigned to 
        self.selected_mod to preserve the current context.
        """

        selection = self._graphical_view.scene().selectedItems()

        if len(selection) == 0:
            if self._selected_mod is not None and self._selected_mod.grouped:
                self._selected_mod = None
        elif len(selection) == 1:
            if isinstance(selection[0], Module):
                if self._selected_mod is not None and self._selected_mod != selection[0]:
                    self._selected_mod.regroup()

                    z_value = self._selected_mod.zValue()+10
                    for branch in selection[0].branches:
                        branch.setZValue(z_value)
                    selection[0].trunk.setZValue(z_value)
                    selection[0].setZValue(z_value) # move it to the front

                self._selected_mod = selection[0]

        self._update_fly_lines_visibility(self._fly_lines_button_group.checkedButton()) # update fly lines based on the selection and the buttons   
        self._update_info_box()

    def create_and_add_module(self, name: str, x: float, y: float, w: float, h: float, atr: str) -> None:
        """
        Creates the main rectangle of a module at position (x,y) with the given width and height.
        A Module that wraps the rectangle is created and added to the graphical view so it becomes visible.

        This function is used to connect the FloorplanDesigner widget to the CreateModule widget.
        """

        if name in self._modules.keys():
            raise KeyError(f"A module named '{name}' already exists.")
        elif name == "":
            raise KeyError(f"The module must have a name.")

        scene_rect = self._graphical_view.sceneRect()
        if w > scene_rect.width() or h > scene_rect.height():
            raise ValueError("Invalid dimensions.")
        elif not(0 < x <= scene_rect.width() and 0 < y <= scene_rect.height()):
            raise ValueError("The point must be inside the scene.")
        
        rect = RectObj(x,y,w,h)
        mod = Module(name, atr, rect)
        self.add_module(mod)

    def add_module(self, module: Module) -> None:
        """A given module is added to the graphical view so it becomes visible. It is also added
        to the widget's internal module set."""
        self._modules[module.name] = module
        self._graphical_view.show_item(module)

    def load_modules(self) -> None:
        """
        Adds the modules of the netlist to the floor widget and the graphical view.
        Note: in PySide6 the Y-axis origin (0,0) is at the top-left corner and increases downwards. The coordinates
        are converted so that the visual representation matches the original positive-up Y-axis.
        """
        colors: list[tuple[float, float, float]] = distinctipy.get_colors(self._netlist.num_modules, pastel_factor=0.4, rng=100) # type: ignore
        i = 0
        scene_h = self._graphical_view.sceneRect().height() # to apply the conversion for the Y-axis
        
        for mod_net in sorted(self._netlist.modules, key=lambda mod: mod.name):
            # if not mod_net.is_iopin:
                # Attribute
                if mod_net.is_hard:
                    atr = "fixed" if mod_net.is_fixed else "hard"
                else:
                    atr = "soft"

                # Create the module with the trunk
                trunk = mod_net.rectangles[0]
                iopin = mod_net.is_iopin
                module_new = Module(mod_net.name, atr, rectangle_to_rectobj(trunk, scene_h), iopin, QColor.fromRgbF(colors[i][0], colors[i][1], colors[i][2]))
                module_new.set_blockages(self._blockages)

                # Add the branches
                for branch in mod_net.rectangles[1:]:
                    module_new.add_rect_to_module(rectangle_to_rectobj(branch, scene_h))

                self.add_module(module_new)
                i += 1

    def save_modules(self) -> None:
        """
        Modifies the netlist with the current module definitions and saves it in a YAML file.
        Converts the y-coordinates back to a positive-up orientation.
        """
        scene_h = self._graphical_view.scene().height() # to apply the conversion for the Y-axis

        for name, module in self._modules.items():
            rects =  list[Rectangle]()
            rects.append(rectobj_to_rectangle(module.trunk, scene_h))
            for branch in module.branches:
                rects.append(rectobj_to_rectangle(branch, scene_h))
            self._netlist.assign_rectangles_module(name, iter(rects))

        self._netlist.write_yaml("tools\\floor_designer\\Examples\\saved_netlist.yml")

    def _ask_how_to_save(self) -> None:

        self.save_modules()

        # msg_box = QMessageBox(self)
        # msg_box.setWindowTitle("Saving Format")
        # msg_box.setText("How would you like to represent the rectangles?\n\n"
        #                 "- Point (x, y), width and height\n"
        #                 "- Top-Left and Bottom-Right corners")
        
        # btn_w_h = QPushButton("Point + w + h", )
        # btn_top_left_bottom_right = QPushButton("T-Left + B-Right")

        # msg_box.addButton(QMessageBox.StandardButton.Cancel)
        # msg_box.addButton(btn_w_h, QMessageBox.ButtonRole.AcceptRole, )
        # msg_box.addButton(btn_top_left_bottom_right, QMessageBox.ButtonRole.AcceptRole)

        # msg_box.exec_()

        # if msg_box.clickedButton() == btn_w_h:
        #     self.save_modules(False)
        # elif msg_box.clickedButton() == btn_top_left_bottom_right:
        #     self.save_modules()

    def _create_info_box(self) -> QGroupBox:
        """Creates and returns a group box to display the selected module information and
        stores its layout in self._info_layout for later updates."""

        info_box = QGroupBox("Selected Module Info")
        info_box.setFixedWidth(240)

        self._info_layout = QVBoxLayout()
        self._info_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._info_layout.setContentsMargins(20, 10, 10, 10)
        self._info_layout.setSpacing(10)
        info_box.setLayout(self._info_layout)

        return info_box
    
    def _update_info_box(self) -> None:
        """Refreshes the info box with data from the selected module."""

        clear_layout(self._info_layout) # Clear old info

        if self._selected_mod is None:
            label = QLabel("No module is currently selected.")
            self._info_layout.addWidget(label)
        else:
            name_label = QLabel(f"<b>Name:</b> {self._selected_mod.name}")
            atr_label = QLabel(f"<b> Attribute:</b> {self._selected_mod.attribute}")

            group_checkbox = QCheckBox("Grouped")
            group_checkbox.setChecked(self._selected_mod.grouped)
            group_checkbox.checkStateChanged.connect(self._group_checkbox_changed)

            self._info_layout.addWidget(name_label)
            self._info_layout.addWidget(atr_label)
            self._info_layout.addWidget(group_checkbox)

            if self._selected_mod.attribute == "soft" and not self._selected_mod.is_iopin:
                area_checkbox = QCheckBox("Maintain area")
                area_checkbox.setChecked(self._selected_mod.maintain_area)
                area_checkbox.checkStateChanged.connect(self._area_checkbox_changed)
                self._info_layout.addWidget(area_checkbox)

            self._info_layout.addWidget(QLabel("<b>Connected to:</b>"))
            label = QLabel(", ".join(self._selected_mod.connected_modules()))
            label.setWordWrap(True) # Break text into lines if it's too long
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
            label.setAlignment(Qt.AlignmentFlag.AlignTop)
            self._info_layout.addWidget(label)
    
    def _group_checkbox_changed(self, state: int) -> None:
        """Groups or ungroups the selected module when the group checkbox changes."""

        if self._selected_mod is not None:
            if state == Qt.CheckState.Unchecked:
                self._selected_mod.ungroup()
            elif state == Qt.CheckState.Checked:
                self._selected_mod.regroup()
                self._selected_mod.setSelected(True)

    def _area_checkbox_changed(self, state: int) -> None:
        """Updates the module's maintain_area attribute when the checkbox changes. 
        This attribute is used during resizing."""

        if self._selected_mod is not None:
            if state == Qt.CheckState.Unchecked:
                self._selected_mod.maintain_area = False
            elif state == Qt.CheckState.Checked:
                self._selected_mod.maintain_area = True

    def load_fly_lines(self) -> None:
        """Initializes the fly lines by reading them from the netlist."""
        self._fly_lines = set[FlyLine]()
        sorted_edges = sorted(self._netlist.edges[:], key= lambda edge: edge.weight)
        if sorted_edges:
            min_weight, max_weight = sorted_edges[0].weight, sorted_edges[-1].weight

        for edge in self._netlist.edges:
            connections = set[Module]() # Modules this fly line connects
            for original_mod in edge.modules:
                floor_mod = self._modules.get(original_mod.name, None)
                if floor_mod is not None: # make sure this module exists in the floor widget
                    connections.add(floor_mod)

            if len(connections) == len(edge.modules): # Doesn't connect terminals
                pen_width = self._scale_weight_to_pen(edge.weight, min_weight, max_weight) # type: ignore
                self._fly_lines.add(FlyLine(self._graphical_view, connections, edge.weight, pen_width))          

    def _scale_weight_to_pen(self, weight: float, min_weight: float, max_weight: float) -> float:
        """ Maps a fly line weight to a pen width:
        - min_weight to MIN_NET_PEN
        - max_weight to MAX_NET_PEN
        - values in between are scaled linearly

        Returns the computed pen width."""

        if min_weight == max_weight:
            return (MIN_NET_PEN + MAX_NET_PEN) / 2
        
        return MIN_NET_PEN + (weight - min_weight) * (MAX_NET_PEN - MIN_NET_PEN) / (max_weight - min_weight)

    def _create_fly_lines_box(self) -> QGroupBox:
        """
        Creates and returns a group box with radio buttons to control fly line visibilitiy.
        Options are:
        - Show all fly lines
        - Show only the fly lines of the selected module
        - Hide all fly lines
        """
        fly_lines_box = QGroupBox("Fly Lines")
        fly_lines_box.setFixedHeight(120)
        fly_lines_box.setFixedWidth(240)

        all_fly_lines_btn = QRadioButton("Show all fly lines")
        module_fly_lines_btn = QRadioButton("Show fly lines of the selected module")
        hide_fly_lines_btn = QRadioButton("Hide all fly lines")

        # Group the radio buttons and put them in a layout
        self._fly_lines_button_group = QButtonGroup(fly_lines_box)
        layout = QVBoxLayout()
        for button in (all_fly_lines_btn, module_fly_lines_btn, hide_fly_lines_btn):
            self._fly_lines_button_group.addButton(button)
            layout.addWidget(button) 

        self._fly_lines_button_group.setExclusive(True)
        self._fly_lines_button_group.buttonClicked.connect(self._update_fly_lines_visibility)
        all_fly_lines_btn.setChecked(True) # Default selection

        fly_lines_box.setLayout(layout)
        return fly_lines_box
    
    def _update_fly_lines_visibility(self, button: QAbstractButton) -> None:
        """Updates fly lines visibility based on the selected radio button option and 
        the currently selected module."""
        
        if button.text() == "Show all fly lines":
            for fly_line in self._fly_lines:
                fly_line.show()
        elif button.text() == "Hide all fly lines" or self._selected_mod is None:
            for fly_line in self._fly_lines:
                fly_line.hide()
        else:
            for fly_line in self._fly_lines:
                fly_line.show() if self._selected_mod.has_fly_line(fly_line) else fly_line.hide()
   
    def _create_visual_details_box(self) -> QGroupBox:
        """
        Creates and returns a group box with checkboxes for visual options.
        - 'Show module names': display the names of the modules
        - 'Mark trunk': show the trunk point of the modules
        """
        visual_details_box = QGroupBox("Visual Details")
        visual_details_box.setFixedHeight(100)
        visual_details_box.setFixedWidth(240)

        names_checkbox = QCheckBox("Show module names")
        names_checkbox.setChecked(True)
        names_checkbox.checkStateChanged.connect(self._names_checkbox_changed)

        trunks_checkbox = QCheckBox("Mark trunks")
        trunks_checkbox.setChecked(True)
        trunks_checkbox.checkStateChanged.connect(self._trunks_checkbox_changed)

        v_layout = QVBoxLayout()
        v_layout.addWidget(names_checkbox)
        v_layout.addWidget(trunks_checkbox)

        visual_details_box.setLayout(v_layout)
        return visual_details_box

    def _names_checkbox_changed(self, state: int) -> None:
        """Shows or hides the names of the modules when the group checkbox changes."""
        show_name = state == Qt.CheckState.Checked

        for module in self._modules.values():
            module.trunk.show_text(show_name)

    def _trunks_checkbox_changed(self, state: int) -> None:
        """Updates the module's _show_trunk_point attribute when the checkbox changes. This attribute 
        controls whether a point is painted in the middle of the trunk."""
        show_trunk_point = state == Qt.CheckState.Checked

        for module in self._modules.values():
            module.trunk.show_trunk_point(show_trunk_point)

    def load_blockages(self, die: Die) -> None:
        """Adds the blockages described in a given die to the graphical view."""
        scene_h = self._graphical_view.scene().height()
        for block in die.blockages:
            new_blockage = rectangle_to_blockage(block, scene_h)
            self._graphical_view.show_item(new_blockage)
            self._blockages.add(new_blockage)

    def scene(self) -> QGraphicsScene:
        """Returns the scene of the widget."""
        return self._graphical_view.scene()

    def remove_selected_item(self) -> str|None:
        """Removes the currently selected item, returning its name if it's a Module, or None otherwise."""
        scene = self._graphical_view.scene()
        for item in scene.selectedItems():
            if isinstance(item, Module):
                del self._modules[item.name]
                scene.removeItem(item)
                return item.name
            if isinstance(item, RectObj):
                selected_mod = self._selected_mod
                assert selected_mod is not None
                if item == selected_mod.trunk:
                    pass
                else:
                    selected_mod.branches.discard(item)
                    selected_mod.update_area()
                    scene.removeItem(item)
        return None

    def get_module(self, module_name: str) -> Module:
        """Given the name of a Module, returns its instance."""
        return self._modules[module_name]

    def group_modules(self) -> None:
        """Ensures all modules in the scene are grouped."""
        for module in self._modules.values():
            if not module.grouped:
                module.regroup()

    def module_names(self) -> Iterator[str]:
        """Yiedls all module names, excluding I/O pins."""
        for name, module in self._modules.items():
            if not module.is_iopin:
                yield name

    def blockages(self) -> Iterator[QGraphicsRectItem]:
        for blockage in self._blockages:
            yield blockage

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._graphical_view.fit_scene()
        return super().resizeEvent(event)

class CreateModule(QWidget):
    """
    A widget that allows the user to create a new module by:
    - Entering the module name
    - Entering rectangle parameters (x, y, width, height) for the module's trunk
    - Defining the module's attributes
    
    Once created, the module is inserted into the given FloorplanDesigner widget.
    """

    _floor: FloorplanDesigner
    _linedit_name: QLineEdit
    _linedit_x: QLineEdit
    _linedit_y: QLineEdit
    _linedit_w: QLineEdit
    _linedit_h: QLineEdit
    _atr_box: QGroupBox

    def __init__(self, floor: FloorplanDesigner):
        super().__init__()

        self._floor = floor

        button = QPushButton("Add")
        button.clicked.connect(self._add_module_to_floor)
        button.setFixedWidth(80)

        general_layout = QVBoxLayout()
        general_layout.addWidget(self._name_box(), stretch=1)
        general_layout.addWidget(self._data_box(), stretch=2)
        self._atr_box = self._attribute_box()
        general_layout.addWidget(self._atr_box, stretch=2)
        general_layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignRight)
        
        general_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setLayout(general_layout)

    def _name_box(self) -> QGroupBox:
        """Creates a group box to introduce the name of the new module."""
        self._linedit_name = QLineEdit()
        self._linedit_name.setFixedWidth(220)

        name_box = QGroupBox("Name of the module")
        layout = QHBoxLayout()
        layout.addWidget(QLabel("name:"))
        layout.addWidget(self._linedit_name)
        layout.addStretch()
        name_box.setLayout(layout)

        return name_box

    def _data_box(self) -> QGroupBox:
        """Creates a group box to introduce the dimensions (width and height) 
        and the initial position of the new module (x,y) which indicates the
        top left corner of the rectangle."""
        label_x, self._linedit_x = create_labeled_lineedit("x")
        label_y, self._linedit_y = create_labeled_lineedit("y")
        label_w, self._linedit_w = create_labeled_lineedit("width")
        label_h, self._linedit_h = create_labeled_lineedit("height")

        grid_layout_xy = create_grid(
            label_x, self._linedit_x, label_y, self._linedit_y
        )
        grid_layout_wh = create_grid(
            label_w, self._linedit_w, label_h, self._linedit_h
        )

        data_box = QGroupBox("Main Rectangle Dimensions")

        data_layout = QHBoxLayout()
        data_layout.addLayout(grid_layout_xy)
        data_layout.addLayout(grid_layout_wh)

        general_layout = QVBoxLayout()
        general_layout.addWidget(QLabel("* (x,y) indicates the top-left point of the rectangle"), Qt.AlignmentFlag.AlignTop)
        general_layout.addLayout(data_layout)

        data_box.setLayout(general_layout)

        return data_box
    
    def _attribute_box(self) -> QGroupBox:
        """Creates a group box to select the attribute of the new module."""
        atr_box = QGroupBox("Choose an attribute")
        soft = QCheckBox("Soft")
        hard = QCheckBox("Hard")
        fixed = QCheckBox("Fixed")
        soft.setChecked(True)

        #Make the checkboxes exclusive
        exclusive_button_group = QButtonGroup(self)
        atr_layout = QVBoxLayout()

        for button in (soft, hard, fixed):
            exclusive_button_group.addButton(button)
            atr_layout.addWidget(button)
            
        exclusive_button_group.setExclusive(True)
        atr_box.setLayout(atr_layout)

        return atr_box

    def _add_module_to_floor(self):
        """Adds the module to the floor widget and clears all the edit lines. Shows
        error messages if inputs are invalid."""
        try:
            name = self._linedit_name.text()
            
            # Attribute
            atr = "soft"
            for child in self._atr_box.children():
                if isinstance(child, QCheckBox):
                    if child.isChecked():
                        atr = child.text().lower()

            self._floor.create_and_add_module(
                name,
                float(self._linedit_x.text()),
                float(self._linedit_y.text()),
                float(self._linedit_w.text()),
                float(self._linedit_h.text()),
                atr
            )
        
            widget3 = self.parent().findChild(CreateRectangle)
            assert widget3 is not None
            widget3.add_item_to_module_combobox(name)

        except KeyError as e:
            QMessageBox.critical(self, "Invalid Module Name",
                                e.args[0],
                                QMessageBox.StandardButton.Ok)           
        except ValueError as e:
            if str(e) == "Invalid dimensions." or str(e) == "The point must be inside the scene.":
                QMessageBox.critical(self, "Invalid Rectangle Size",
                                e.args[0],
                                QMessageBox.StandardButton.Ok)
            else:
                QMessageBox.critical(self, "Invalid Rectangle Data",
                                    "Please enter valid numbers.",
                                    QMessageBox.StandardButton.Ok)

        self._linedit_name.clear()
        self._linedit_x.clear()
        self._linedit_y.clear()
        self._linedit_w.clear()
        self._linedit_h.clear()


class CreateRectangle(QWidget):
    """
    A widget used to add rectangles to a specific module. 
    Users can select a module, add rectangles to it, and then submit these new rectangles
    to be added to the floor (widget1).
    """
    _floor: FloorplanDesigner
    _tab_widget: QTabWidget
    _graphical_view: GraphicalView # Not the same instance as widget 1 (floor)
    _module_combo_box: QComboBox # To select a module
    _selected_module: Module
    _new_rects: set[RectObj] # New rectangles that will be added to the module
    _add_to_design_button: QPushButton

    def __init__(self, floor: FloorplanDesigner, tab_widget: QTabWidget):
        super().__init__()
        self._floor = floor
        self._tab_widget = tab_widget
        self._tab_widget.currentChanged.connect(self._tab_changed) # Updates the thrid tab if it's necessary
        self._new_rects = set[RectObj]()

        dim_act_layout = QVBoxLayout()
        dim_act_layout.addWidget(self._choose_module_box())
        dim_act_layout.addWidget(self._dimensions_box())
        dim_act_layout.addWidget(self._actions_box())

        h_layout = QHBoxLayout()
        scene_rect = self._floor.scene().sceneRect()
        self._graphical_view = GraphicalView(int(scene_rect.width()), int(scene_rect.height()))
        h_layout.addWidget(self._graphical_view)
        h_layout.addLayout(dim_act_layout)
        h_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.setLayout(h_layout)
        
        self._module_combo_box.currentTextChanged.connect(self._clone_scene_as_background)
        self._clone_scene_as_background()

    def _choose_module_box(self) -> QGroupBox:
        """Creates a group box for selecting a module to add the rectangle.
        Returns : The group box with a combo box."""

        mod_box = QGroupBox("Select Module")
        mod_box.setFixedWidth(280)
        label = QLabel("Which module would you like to add the rectangle to?")
        self._module_combo_box = QComboBox()
        self._module_combo_box.setInsertPolicy(QComboBox.InsertPolicy.InsertAlphabetically)

        for module_name in self._floor.module_names():
            self._module_combo_box.addItem(module_name)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self._module_combo_box)
        mod_box.setLayout(layout)
        mod_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        return mod_box
    
    def _dimensions_box(self) -> QGroupBox:
        """Creates a group box to introduce the dimensions (width and height) of a new rectangle.
        Returns : The group box with the labels, lines to edit and a button."""

        dim_box = QGroupBox("Dimensions")
        dim_box.setFixedWidth(280)
        label_w, self._linedit_w = create_labeled_lineedit("width")
        label_h, self._linedit_h = create_labeled_lineedit("height")
        grid = create_grid(label_w, self._linedit_w, label_h, self._linedit_h)
        
        add_button = QPushButton("Add")
        add_button.setFixedSize(QSize(40,20))
        add_button.clicked.connect(self._add_provisional_rect)

        layout = QVBoxLayout()
        layout.addLayout(grid)
        layout.addWidget(add_button, alignment=Qt.AlignmentFlag.AlignRight)

        dim_box.setLayout(layout)
        dim_box.setFixedHeight(140)

        return dim_box
    
    def _actions_box(self) -> QGroupBox:
        """Creates a group box containing two buttons:
        - Join : joins the new rectangles to the trunk
        - Add to Design : adds the new rectangles to the floorplan design on the first tab.

        Returns : The group box with the action buttons.
        """

        act_box = QGroupBox("Actions")
        act_box.setFixedWidth(280)

        self._add_to_design_button = QPushButton("Add to Design")
        self._add_to_design_button.setEnabled(False)
        self._add_to_design_button.clicked.connect(self._add_to_floor_scene)

        join_button = QPushButton("Join")
        join_button.clicked.connect(lambda checked=False: self._join_new_rectangles(self._add_to_design_button))

        layout = QVBoxLayout()
        layout.addWidget(join_button)
        layout.addWidget(self._add_to_design_button)

        act_box.setLayout(layout)

        return act_box
    
    def _add_provisional_rect(self) -> None:
        """
        Adds a new provisional rectangle to the currently selected module. The rectangle is created 
        using the width and height entered in the line edits. It is added to the module, but immediately 
        removed from the QGraphicsItemGroup so it can be moved independently.

        The 'Add to Design' button is disabled and the edit lines are cleared. If the input data is 
        invalid, an error message is shown.
        """

        try:
            rect = RectObj(10, 10, float(self._linedit_w.text()), float(self._linedit_h.text()))
            rect.setBrush(self._selected_module.trunk.brush())

            self._new_rects.add(rect)
            self._selected_module.add_rect_to_module(rect)
            self._selected_module.removeFromGroup(rect) # to be able to move it
            self._add_to_design_button.setEnabled(False)

            rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)

            self._selected_module.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
            self._selected_module.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        except:
            QMessageBox.critical(self, "Invalid Rectangle Data",
                                "Please enter valid numbers.",
                                QMessageBox.StandardButton.Ok)
            
        self._linedit_w.clear()
        self._linedit_h.clear()
    
    def _join_new_rectangles(self, add_to_design_button: QPushButton) -> None:
        """Joins the rectangles of the currently selected module and enables the 'Add to
        Design' button."""

        self._selected_module.join()
        add_to_design_button.setEnabled(True)

    def _add_to_floor_scene(self) -> None:
        """Adds the new rectangles to the corresponding module in the floorplan design.
        Each rectangles is copied and added to the selected module on the first tab.
        The set of new rectangles is cleared, and the current tab is switched to the
        floorplan designer."""

        module = self._floor.get_module(self._module_combo_box.currentText())
        for rect in self._new_rects:
            module.add_rect_to_module(rect.copy())

        self._new_rects.clear()
        self._tab_widget.setCurrentWidget(self._floor)

    def add_item_to_module_combobox(self, item: str) -> None:
        """Adds an item to the combo box of the widget."""
        self._module_combo_box.addItem(item)

    def _clone_scene_as_background(self):
        """
        Clears the current scene and clones all the Module items from the floor's graphical view.
        All the modules are made non-selectable and non-movable. The currently selected module is
        assigned to the 'selected module' attribute and keeps its color, while the rest are shown 
        in gray.
        This function also copies the blockages.
        """
        self._graphical_view.clear_scene()
        
        for item in self._floor.scene().items():
            if isinstance(item, Module):
                if not item.is_iopin:
                    clone = item.copy()
                    clone.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                    clone.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)

                    clone.trunk.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                    for rect in clone.branches:
                        rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

                    self._graphical_view.show_item(clone)

                    if item.name == self._module_combo_box.currentText():
                        self._selected_module = clone
                    else:
                        clone.setBrush(QBrush(Qt.GlobalColor.lightGray))
                        clone.setZValue(-1)
            
            for item in self._floor.blockages():
                blockage = QGraphicsRectItem(item.rect())
                blockage.setPos(item.pos())
                set_blockage_style(blockage)
                self._graphical_view.show_item(blockage)


        self._new_rects.clear()

    def _tab_changed(self, index: int) -> None:
        """Updates the scene when this tab ('Add Rectangle To A Module') is selected.
        Groups the modules of the first tab if any are ungrouped."""
        if index == self._tab_widget.indexOf(self):
            self._floor.group_modules() # Make sure all the modules are grouped
            # Prepare the scene of the third tab
            self._clone_scene_as_background()

    def remove_from_combobox(self, module_name: str) -> None:
        """Removes the given module name from the combobox."""
        index = self._module_combo_box.findText(module_name) # Find the item index in the combo box to delete it
        if index != -1:
            self._module_combo_box.removeItem(index)

    def remove_selected_items(self) -> None:
        """Removes the currently selected rectangles."""
        scene = self._graphical_view.scene()
        for item in scene.selectedItems():
            # Remove rectangles that were recently added in this tab
            if isinstance(item, RectObj) and item in self._new_rects:
                self._new_rects.remove(item)
                scene.removeItem(item)

#### Other functions:
def create_labeled_lineedit(label_text: str) -> tuple[QLabel, QLineEdit]:
    """Creates a label with the given text and a styled line edit. Returns them as a tuple."""
    label = QLabel(f"{label_text} : ")
    line_edit = QLineEdit()
    line_edit.setPlaceholderText("0")
    line_edit.setFixedWidth(60)
    line_edit.setStyleSheet(
        "padding: 4px; border: 1px solid #ccc; border-radius: 4px;"
    )

    return label, line_edit

def create_grid(a_label: QLabel, a_line: QLineEdit, b_label: QLabel, b_line: QLineEdit) -> QGridLayout:
    """Creates a 2-row grid layout for positioning two labels and their corresponding line edits."""
    grid_layout = QGridLayout()
    grid_layout.addWidget(a_label, 0, 0, Qt.AlignmentFlag.AlignRight)
    grid_layout.addWidget(a_line, 0, 1, Qt.AlignmentFlag.AlignLeft)
    grid_layout.addWidget(b_label, 1, 0, Qt.AlignmentFlag.AlignRight)
    grid_layout.addWidget(b_line, 1, 1, Qt.AlignmentFlag.AlignLeft)

    return grid_layout

def clear_layout(layout: QLayout) -> None:
    """Removes and deletes all widgets from the given layout."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()

        if widget:
            widget.deleteLater()

def rectangle_to_rectobj(rect: Rectangle, scene_h: float) -> RectObj:
    """
    Converts a Rectangle (defined by center and shape) into a RectObj (defined by top-left corner
    and size).
    Flips the Y-axis for PySide6's positive-down orientation using the given scene height.
    """

    center = rect.center
    w, h = rect.shape.w, rect.shape.h
    return RectObj(center.x - w/2, scene_h - center.y - h/2, w, h)
    
def rectobj_to_rectangle(rectobj: RectObj, scene_h: float) -> Rectangle:
    """
    Creates Rectangle from a given RectObj.
    Flips the Y-axis back to positive-up orientation using the given scene height.
    Returns the new Rectangle.
    """
    rectobj_rect = rectobj.rect()
    w, h = rectobj_rect.width(), rectobj_rect.height()
    if rectobj.is_iopin:
        if w == 0.1:
            w = 0
        if h == 0.1:
            h = 0

        assert w == 0 or h == 0

    pos = rectobj.scenePos()
    x, y = pos.x(), pos.y() # top-left corner
    scene_h = scene_h

    return Rectangle(center=Point(x + w/2, scene_h - y - h/2), shape=Shape(w,h))

def rectangle_to_blockage(rect: Rectangle, scene_h: float) -> QGraphicsRectItem:
    """
    Converts a Rectangle (defined by center and shape) into a blockage (defined by top-left corner
    and size).
    Flips the Y-axis for PySide6's positive-down orientation using the given scene height.
    """

    center = rect.center
    w, h = rect.shape.w, rect.shape.h

    new_blockage = QGraphicsRectItem(0, 0, w, h)
    new_blockage.setPos(center.x - w/2, scene_h - center.y - h/2)

    set_blockage_style(new_blockage)

    return new_blockage

def set_blockage_style(blockage: QGraphicsRectItem) -> None:
    """Colors the given rectangle gray and fills it with diagonal lines."""
    brush = QBrush()
    # brush.setColor(Qt.GlobalColor.gray)
    brush.setStyle(Qt.BrushStyle.FDiagPattern)
    blockage.setBrush(brush)
    pen = blockage.pen()
    pen.setCosmetic(True)
    blockage.setPen(pen)
