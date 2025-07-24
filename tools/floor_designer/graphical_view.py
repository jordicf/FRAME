from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsItem,
    QFrame
)
from PySide6.QtCore import Qt

class GraphicalView(QGraphicsView):
    """Class that provides a widget for displaying the contents of a QGraphicsScene."""
    scene_ref: QGraphicsScene

    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.scene_ref = QGraphicsScene()
        self.setScene(self.scene_ref)
        self.scene_ref.setSceneRect(0, 0, 400, 400)

    def show_item(self, item: QGraphicsItem) -> None:
        """Adds an item to the graphical view so it becomes visible on the screen."""
        self.scene_ref.addItem(item)

    def clear_scene(self) -> None:
        
        self.scene_ref.clear()
            
