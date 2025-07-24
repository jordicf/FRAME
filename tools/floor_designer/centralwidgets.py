import json
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QLineEdit,
    QGridLayout,
    QGroupBox,
    QCheckBox,
    QButtonGroup,
    QMessageBox,
    QComboBox,
    QGraphicsItem,
    QTabWidget,
    QLayout
)
from PySide6.QtCore import Qt, QSize
from rectangles import RectObj
from graphical_view import GraphicalView
from module import Module, QBrush

class FloorplanDesigner(QWidget):
    """A widget for designing chip floorplans by placing and managing rectangular modules."""

    graphical_view: GraphicalView
    modules: dict[str,Module]
    selected_mod: Module|None = None
    _info_layout: QVBoxLayout

    def __init__(self):
        super().__init__()

        self.graphical_view = GraphicalView()
        self.modules = dict[str,Module]()
        self.load_modules()
        self.graphical_view.scene().selectionChanged.connect(self._selected_module_changed)

        save_button = QPushButton("Save")
        save_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        save_button.clicked.connect(self.ask_how_to_save)

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.graphical_view)
        h_layout.addWidget(self._create_info_box(), alignment=Qt.AlignmentFlag.AlignRight)
        self._update_info_box()

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
        """"""

        selection = self.graphical_view.scene().selectedItems()

        if len(selection) == 0:
            if self.selected_mod is not None and self.selected_mod.grouped:
                self.selected_mod = None
        elif len(selection) == 1:
            if isinstance(selection[0], Module):
                if self.selected_mod is not None and self.selected_mod != selection[0]:
                    self.selected_mod.regroup()

                self.selected_mod = selection[0]

        self._update_info_box()

    def create_and_add_module(self, name: str, x: float, y: float, w: float, h: float, atr: str) -> None:
        """Creates the main rectangle of a module at position (x,y) with the given width and height.
        A Module that wraps the rectangle is created and added to the graphical view
        so it becomes visible. """
        rect = RectObj(x,y,w,h)
        mod = Module(name, atr, rect)
        self.add_module(mod)

    def add_module(self, module: Module) -> None:
        """A given module is added to the graphical view so it becomes visible. It is also added
        to the widget's internal module set."""
        self.modules[module.name] = module
        self.graphical_view.show_item(module)

    def load_modules(self) -> None:
        """Loads modules from a JSON file and adds them to the graphical view."""

        with open("rects.json") as f:
            data = json.load(f)

        for module in data["modules"]:

            main_rect = self.dict_to_rect(module["rectangles"][0])
            mod = Module(module["name"], module["attribute"], main_rect)

            for rect in module["rectangles"][1:]:
                rect_obj = self.dict_to_rect(rect)
                mod.add_rect_to_module(rect_obj)

            self.add_module(mod)

    def dict_to_rect(self, d_rect: dict[str,float]) -> RectObj:
        
        return RectObj(float(d_rect["x"]), float(d_rect["y"]), float(d_rect["w"]), float(d_rect["h"]))

    def save_modules(self, bounding_rect: bool = True) -> None:

        data = {}
        data["modules"] = []
        for mod in self.modules.values():
            mod_dict = {}
            mod_dict["name"] = mod.name
            mod_dict["attribute"] =  mod.attribute
            
            rects = list[dict[str,float]]()
            for rect in [mod.trunk] + list(mod.branches):
                x, y = rect.scenePos().x(), rect.scenePos().y()
                w, h = rect.rect().width(), rect.rect().height()

                if bounding_rect:

                    rect_dict = {
                        "tl_x" : x,
                        "tl_y" : y,
                        "br_x" : x + w,
                        "br_y" : y + h
                    }
                else:
                    rect_dict = {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h
                    }
                    
                rects.append(rect_dict)

            mod_dict["rectangles"] = rects
            data["modules"].append(mod_dict) # type: ignore

        file_name = "bounding_rect" if bounding_rect else "width_height"

        with open(f"new_rects_{file_name}.json", "w") as f:
            json.dump(data, f, indent=2)

    def ask_how_to_save(self) -> None:

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Saving Format")
        msg_box.setText("How would you like to represent the rectangles?\n\n"
                        "- Point (x, y), width and height\n"
                        "- Top-Left and Bottom-Right corners")
        
        btn_w_h = QPushButton("Point + w + h", )
        btn_top_left_bottom_right = QPushButton("T-Left + B-Right")

        msg_box.addButton(QMessageBox.StandardButton.Cancel)
        msg_box.addButton(btn_w_h, QMessageBox.ButtonRole.AcceptRole, )
        msg_box.addButton(btn_top_left_bottom_right, QMessageBox.ButtonRole.AcceptRole)

        msg_box.exec_()

        if msg_box.clickedButton() == btn_w_h:
            self.save_modules(False)
        elif msg_box.clickedButton() == btn_top_left_bottom_right:
            self.save_modules()

    def _create_info_box(self) -> QGroupBox:
        """Creates and returns a group box to display the selected module information and
        stores it's layout in self._info_layout for later updates."""

        info_box = QGroupBox("Selected Module Info")
        info_box.setFixedWidth(260)

        self._info_layout = QVBoxLayout()
        self._info_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._info_layout.setContentsMargins(20, 10, 10, 10)
        self._info_layout.setSpacing(5)
        info_box.setLayout(self._info_layout)

        return info_box
    
    def _update_info_box(self) -> None:
        """Refreshes the info box with data from the selected module."""

        clear_layout(self._info_layout) # Clear old info

        if self.selected_mod is None:
            label = QLabel("No module is currently selected.")
            self._info_layout.addWidget(label)
        else:
            name_label = QLabel(f"<b>Name:</b> {self.selected_mod.name}")
            atr_label = QLabel(f"<b> Attribute:</b> {self.selected_mod.attribute}")

            group_checkbox = QCheckBox("Grouped")
            group_checkbox.setChecked(self.selected_mod.grouped)
            group_checkbox.checkStateChanged.connect(self._state_checkbox_changed)

            self._info_layout.addWidget(name_label)
            self._info_layout.addWidget(atr_label)
            self._info_layout.addWidget(group_checkbox)

    
    def _state_checkbox_changed(self, state: int) -> None:
        """Groups or ungroups the selected module when the checkbox changes."""

        if self.selected_mod is not None:
            if state == Qt.CheckState.Unchecked:
                self.selected_mod.ungroup()
            elif state == Qt.CheckState.Checked:
                self.selected_mod.regroup()
                self.selected_mod.setSelected(True)

class CreateModule(QWidget):
    """A widget that allows the user to input rectangle parameters (x, y, width, height) 
    and insert the resulting rectangle into a given Floor widget."""

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
        button.clicked.connect(self._add_module)
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
        """Creates a group box to select the attribute of the new rectangle."""
        atr_box = QGroupBox("Choose an attribute")
        soft = QCheckBox("Soft")
        hard = QCheckBox("Hard")
        fixed = QCheckBox("Fixed")
        soft.setChecked(True)

        #Make the checkboxes exclusive
        exclusive_button_group = QButtonGroup(self)
        for button in (soft, hard, fixed):
            exclusive_button_group.addButton(button)
        exclusive_button_group.setExclusive(True)

        atr_layout = QVBoxLayout()
        atr_layout.addWidget(soft)
        atr_layout.addWidget(hard)
        atr_layout.addWidget(fixed)
        atr_box.setLayout(atr_layout)

        return atr_box

    def _add_module(self):
        """Adds the module to the floor widget and clears all the edit lines."""
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
        except:

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
    A Widget used to add rectangles to a specific module.
    """
    _floor: FloorplanDesigner
    _tab_widget: QTabWidget
    graphical_view: GraphicalView
    module_combo_box: QComboBox
    selected_module: Module
    new_rects: set[RectObj]

    def __init__(self, floor: FloorplanDesigner, tab_widget: QTabWidget):
        super().__init__()
        self._floor = floor
        self._tab_widget = tab_widget
        self._tab_widget.currentChanged.connect(self._tab_changed) # Updates the thrid tab if it's necessary
        self.new_rects = set[RectObj]()

        v_layout = QVBoxLayout()
        v_layout.addWidget(self._choose_module_box())

        dim_act_layout = QVBoxLayout()
        dim_act_layout.addWidget(self._dimensions_box())
        dim_act_layout.addWidget(self._actions_box())

        h_layout = QHBoxLayout()
        self.graphical_view = GraphicalView()
        h_layout.addWidget(self.graphical_view)
        h_layout.addLayout(dim_act_layout)
        h_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        v_layout.addLayout(h_layout)
        self.setLayout(v_layout)
        
        self.module_combo_box.currentTextChanged.connect(self._clone_scene_as_background)
        self._clone_scene_as_background()


    def _choose_module_box(self) -> QGroupBox:
        """Creates a group box for selecting a module to add the rectangle.
        Returns : The group box with a combo box."""

        mod_box = QGroupBox("Select Module")
        label = QLabel("Which module would you like to add the rectangle to?")
        self.module_combo_box = QComboBox()

        for module_name in self._floor.modules.keys():
            self.module_combo_box.addItem(module_name)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.module_combo_box)
        mod_box.setLayout(layout)
        mod_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        return mod_box
    
    def _dimensions_box(self) -> QGroupBox:
        """Creates a group box to introduce the dimensions (width and height) of a new rectangle.
        Returns : The group box with the labels, lines to edit and a button."""

        dim_box = QGroupBox("Dimensions")
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

        join_button = QPushButton("Join")
        join_button.clicked.connect(lambda checked=False: self._join_new_rectangles(self.add_to_design_button))

        self.add_to_design_button = QPushButton("Add to Design")
        self.add_to_design_button.setEnabled(False)
        self.add_to_design_button.clicked.connect(self._add_to_floor_scene)

        layout = QVBoxLayout()
        layout.addWidget(join_button)
        layout.addWidget(self.add_to_design_button)

        act_box.setLayout(layout)

        return act_box
    
    def _add_provisional_rect(self) -> None:
        """Adds a new provisional rectangle to the currently selected module. The rectangle is created 
        using the width and height entered in the line edits. It is added to the module, but immediately 
        removed from the QGraphicsItemGroup so it can be moved independently.
        The 'Add to Design' button is disabled and the edit lines are cleared. If the input data is 
        invalid, an error message is shown."""

        try:
            rect = RectObj(10, 10, float(self._linedit_w.text()), float(self._linedit_h.text()))
            rect.setBrush(self.selected_module.trunk.brush())

            self.graphical_view.show_item(rect)
            self.new_rects.add(rect)
            self.selected_module.add_rect_to_module(rect)
            self.selected_module.removeFromGroup(rect) # to be able to move it
            self.add_to_design_button.setEnabled(False)

            rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)

            self.selected_module.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
            self.selected_module.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        except:
            QMessageBox.critical(self, "Invalid Rectangle Data",
                                "Please enter valid numbers.",
                                QMessageBox.StandardButton.Ok)
            
        self._linedit_w.clear()
        self._linedit_h.clear()
    
    def _join_new_rectangles(self, add_to_design_button: QPushButton) -> None:
        """Joins the rectangles of the currently selected module and enables the 'Add to
        Design' button."""

        self.selected_module.join()
        add_to_design_button.setEnabled(True)

    def _add_to_floor_scene(self) -> None:
        """Adds the new rectangles to the corresponding module in the floorplan design.
        Each rectangles is copied and added to the selected module on the first tab.
        The set of new rectangles is cleared, and the current tab is switched to the
        floorplan designer."""

        for rect in self.new_rects:
            self._floor.modules[self.module_combo_box.currentText()].add_rect_to_module(rect.copy())

        self.new_rects.clear()
        self._tab_widget.setCurrentWidget(self._floor)


    def add_item_to_module_combobox(self, item: str) -> None:
        """Adds an item to the combo box of the widget."""
        self.module_combo_box.addItem(item)

    def _clone_scene_as_background(self):
        """
        Clears the current scene and clones all the Module items from the floor's graphical view.
        All the modules are made non-selectable and non-movable. The currently selected module is
        assigned to the 'selected module' attribute and keeps its color, while the rest are shown 
        in gray.
        """
        self.graphical_view.clear_scene()
        
        for item in self._floor.graphical_view.items():
            if isinstance(item, Module):
                clone = item.copy()
                clone.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                clone.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)

                clone.trunk.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                for rect in clone.branches:
                    rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

                self.graphical_view.show_item(clone)

                if item.name == self.module_combo_box.currentText():
                    self.selected_module = clone
                else:
                    clone.setBrush(QBrush(Qt.GlobalColor.lightGray))
                    clone.setZValue(-1)

        self.new_rects.clear()

    def _tab_changed(self, index: int) -> None:
        """Updates the scene if the 'Add Rectangle To A Module' tab is selected."""
        if index == self._tab_widget.indexOf(self):
            self._clone_scene_as_background()


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

    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget:
            widget.deleteLater()
