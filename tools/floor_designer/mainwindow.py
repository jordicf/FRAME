from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QMainWindow,QToolBar,QPushButton,QStatusBar, QApplication, QTabWidget, QMessageBox
from centralwidgets import FloorplanDesigner, CreateModule, Module, CreateRectangle, RectObj

class MainWindow(QMainWindow):
    """Main application window for the floorplanning tool.
    
    This windows contains tabs with three main tools:
    - A floorlplan designer to interactively design the chip layout.
    - A module creator for adding new modules to the floorplan.
    - A rectangle creator for adding rectangles to existing modules.
    
    It includes a menu, a toolbar and a status bar."""

    _app: QApplication
    _tab_widget: QTabWidget
    _widget1: FloorplanDesigner
    _widget2: CreateModule
    _widget3: CreateRectangle

    def __init__(self, app: QApplication):
        super().__init__()
        self._app  = app
        self.setWindowTitle("Floorplanning")

        self._create_menu()
        self._create_toolbar()
        self.setStatusBar(QStatusBar(self))

        # Create widgets and tabs
        self._tab_widget = QTabWidget()
        self._widget1 = FloorplanDesigner()
        self._widget2 = CreateModule(self._widget1)
        self._widget3 = CreateRectangle(self._widget1, self._tab_widget)

        for widget in (self._widget1, self._widget2, self._widget3):
            widget.setParent(self)

        self._tab_widget.addTab(self._widget1, "Design")
        self._tab_widget.addTab(self._widget2, "Add Module")
        self._tab_widget.addTab(self._widget3, "Add Rectangle To A Module")

        self.setCentralWidget(self._tab_widget)

    def _create_menu(self) -> None:
        """Creates a menu located at the top of the window."""

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        # An acction is an object that you can either add to the toolbar or the menu bar
        quit_action = file_menu.addAction("Quit")
        file_menu.addAction("Save")
        quit_action.triggered.connect(self._app.quit)

        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction("Copy")
        edit_menu.addAction("Cut")
        edit_menu.addAction("Paste")
        edit_menu.addAction("Undo")
        edit_menu.addAction("Redo")

        menu_bar.addMenu("Window")
        menu_bar.addMenu("Settings")
        menu_bar.addMenu("Help")

        self.quit_action = quit_action # To use it in the toolbar

    def _create_toolbar(self) -> None:
        """Creates a toolbar located at the left side of the window."""

        toolbar = QToolBar("Toolbar")
        toolbar.setIconSize(QSize(16,16))
        toolbar.setMovable(False)
        toolbar.setOrientation(Qt.Orientation.Vertical)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, toolbar)

        toolbar.addAction(self.quit_action) # Add the quit action to the toolbar
        self.quit_action.setStatusTip("Exit the application")

        toolbar.addSeparator()

        remove_button = QPushButton("Remove")
        remove_button.setStatusTip("Remove the currently selected item")
        remove_button.clicked.connect(self._remove_selected_item)
        toolbar.addWidget(remove_button)

        rotate_button = QPushButton("Rotate")
        rotate_button.setStatusTip("Rotate the currently selected item")
        rotate_button.clicked.connect(self._rotate_selected_item)
        toolbar.addWidget(rotate_button)

    def _remove_selected_item(self) -> None:
        """Removes the currently selected items from the scene acter user confirmation.
        Modules or rectangles depending on the active tab. In the Add Rectangle tab, you 
        can only delete the new rectangles that are currntly being added."""

        question = QMessageBox.question(self, "Remove items",
                                    "Are you sure you want to remove the selected item?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if question == QMessageBox.StandardButton.Yes:
            current_tab = self._tab_widget.tabText(self._tab_widget.currentIndex())
            if current_tab == "Design":
                scene = self._widget1.graphical_view.scene()
                for item in scene.selectedItems():
                    if isinstance(item, Module):
                        del self._widget1.modules[item.name]
                        # Find the item index in the combo box to delete it
                        index = self._widget3.module_combo_box.findText(item.name)
                        if index != -1:
                            self._widget3.module_combo_box.removeItem(index)
                        scene.removeItem(item)
                    if isinstance(item, RectObj):
                        selected_mod = self._widget1.selected_mod
                        assert selected_mod is not None
                        if item == selected_mod.trunk:
                            pass
                        else:
                            selected_mod.branches.discard(item)
                            scene.removeItem(item)
            elif current_tab == "Add Rectangle To A Module":
                scene = self._widget3.graphical_view.scene() # scene of the third tab
                for item in scene.selectedItems():
                    # Remove rectangles that were recently added in this tab
                    if isinstance(item, RectObj) and item in self._widget3.new_rects:
                        self._widget3.new_rects.remove(item)
                        scene.removeItem(item)
        
    def _rotate_selected_item(self) -> None:
        """Rotate the currently selected item by 90 degrees clockwise."""
        selected_items = self._widget1.graphical_view.scene().selectedItems()

        if len(selected_items) == 1:
            item = selected_items[0]
            if isinstance(item, Module) or isinstance(selected_items[0], RectObj):
                item.setTransformOriginPoint(item.boundingRect().center())
                item.setRotation((item.rotation() + 90) % 360)