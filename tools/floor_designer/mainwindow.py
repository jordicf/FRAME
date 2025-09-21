from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QMainWindow,QToolBar,QPushButton,QStatusBar, QApplication, QTabWidget, QMessageBox, QFileDialog
from centralwidgets import FloorplanDesigner, CreateModule, CreateRectangle

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
        # self._widget2 = CreateModule(self._widget1)
        self._widget3 = CreateRectangle(self._widget1, self._tab_widget)

        for widget in (self._widget1, self._widget3):
            widget.setParent(self)

        self._tab_widget.addTab(self._widget1, "Design")
        # self._tab_widget.addTab(self._widget2, "Add Module")
        self._tab_widget.addTab(self._widget3, "Add Rectangle To A Module")

        self.setCentralWidget(self._tab_widget)

    def _create_menu(self) -> None:
        """Creates a menu located at the top of the window."""

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        # An acction is an object that you can either add to the toolbar or the menu bar
        open_action = file_menu.addAction("Open Design")
        open_action.triggered.connect(self._open_files)
        quit_action = file_menu.addAction("Quit")
        quit_action.triggered.connect(self._app.quit)
        file_menu.addAction("Save")

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
                self._widget1.remove_selected_item()
            elif current_tab == "Add Rectangle To A Module":
                self._widget3.remove_selected_items()
        
    def _open_files(self) -> None:

        die_path, _ = QFileDialog.getOpenFileName(self, "Select Die File", "tools\\floor_designer\\Examples", "YAML/JSON files (*.yaml *.yml *.json);;All Files (*)")
        if not die_path:
            return

        netlist_path, _ = QFileDialog.getOpenFileName(self, "Select Netlist File", "tools\\floor_designer\\Examples", "YAML/JSON files (*.yaml *.yml *.json);;All Files (*)")
        if not netlist_path:
            return
        
        self._widget1.load_files(die_path, netlist_path)
        self._widget3.set_combobox_and_scene()
