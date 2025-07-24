from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow,QToolBar,QPushButton,QStatusBar, QApplication, QTabWidget, QMessageBox
from centralwidgets import FloorplanDesigner, CreateModule, Module, CreateRectangle, RectObj

class MainWindow(QMainWindow):
    _app: QApplication
    _tab_widget: QTabWidget
    _widget1: FloorplanDesigner
    _widget2: CreateModule
    _widget3: CreateRectangle

    def __init__(self, app: QApplication):
        super().__init__()
        self._app  = app
        self.setWindowTitle("Floorplanning")
        self.setFixedSize(QSize(840, 620))

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

        toolbar = QToolBar("My main toolbar")
        toolbar.setIconSize(QSize(16,16))
        toolbar.setMovable(False)
        toolbar.setOrientation(Qt.Orientation.Vertical)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, toolbar)

        # Add the quit action to the toolbar
        toolbar.addAction(self.quit_action)

        action1 = QAction("Some Action", self)
        action1.setStatusTip("Status message for some action")
        action1.triggered.connect(self._toolbar_button_click)
        toolbar.addAction(action1)

        action2 = QAction("Some other action", self)
        action2.setStatusTip("Status message for some other action")
        action2.triggered.connect(self._toolbar_button_click)
        action2.setCheckable(True)
        toolbar.addAction(action2)

        toolbar.addSeparator()
        remove_button = QPushButton("Remove")
        toolbar.addWidget(remove_button)
        remove_button.clicked.connect(self._remove_selected_item)

    def _toolbar_button_click(self) -> None:
        """"""
        self.statusBar().showMessage("Some message ...", 3000)

    def _remove_selected_item(self) -> None:
        
        question = QMessageBox.question(self, "Remove items",
                                    "Are you sure you want to remove the selected item?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if question == QMessageBox.StandardButton.Yes:

            current_tab = self._tab_widget.tabText(self._tab_widget.currentIndex())
            if current_tab == "Design":
                scene = self._widget1.graphical_view.scene()
                for item in scene.selectedItems():
                    if isinstance(item, Module):
                        for rect in item.branches:
                            scene.removeItem(rect)
                        del self._widget1.modules[item.name]
                        # Find the item index in the combo box to delete it
                        index = self._widget3.module_combo_box.findText(item.name)
                        if index != -1:
                            self._widget3.module_combo_box.removeItem(index)
                        scene.removeItem(item) 
            elif current_tab == "Add Rectangle To A Module":
                scene = self._widget3.graphical_view.scene()
                for item in scene.selectedItems():
                    if isinstance(item, RectObj) and item in self._widget3.new_rects:
                        self._widget3.new_rects.remove(item)
                        scene.removeItem(item)
        

