from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsItem,
    QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QWheelEvent, QPainter

class GraphicalView(QGraphicsView):
    """
    Class that manages and displays an internal QGraphicsScene.
    The class creates its own QGrpahicsScene and provides functionallity to visualize
    and interct with it. Supports zooming in and out.
    """
    scene_ref: QGraphicsScene
    _min_zoom: float
    _current_zoom: float

    def __init__(self, scene_width: int = 400, scene_height: int = 400) -> None:
        super().__init__()

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.scene_ref = QGraphicsScene()
        self.setScene(self.scene_ref)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Adjust the view and the scene
        self.scene_ref.setSceneRect(0,0,scene_width, scene_height)
        self._fit_scene()
        self._adjust_view_to_scene()

    def show_item(self, item: QGraphicsItem) -> None:
        """Adds an item to the scene so it becomes visible on the screen."""
        self.scene_ref.addItem(item)

    def clear_scene(self) -> None:
        """Clears all the elements from the scene."""
        self.scene_ref.clear()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Handles mouse wheel events to control zoom. 
        Zooms in if the Ctrl key is pressed and the wheel is rotated forwards, and zooms out if 
        the wheel is rotated backwards.
        """

        if not (event.modifiers() and Qt.Modifier.CTRL):
            event.ignore()
        else:
            wheel_change = event.angleDelta().y()
            if wheel_change > 0: # wheel rotated forwards (away from user)
                self._zoom_in()
            elif wheel_change < 0: # wheel rotated backwards (toward user)
                self._zoom_out()
            event.accept()
        return super().wheelEvent(event)

    def _zoom_in(self) -> None:
        """Increases the scene zoom level by a factor of 1.05."""
        self.scale(1.05, 1.05)
        self._current_zoom += 1.05
    
    def _zoom_out(self) -> None:
        """Decreases the scene zoom level by a factor of 0.8, or fits the scene if the 
        minimum zoom level is reached."""
        current_scale = self.transform().m11()  # Zoom real aplicado
        new_zoom = current_scale * 0.8
        if new_zoom >= self._min_zoom:
            self.scale(0.8, 0.8)
            self._current_zoom = new_zoom
        else:
            self._fit_scene()

    def _fit_scene(self) -> None:
        """Fits the scene rect into the view preserving the aspect ratio, ensuring
        the whole scene is visible. Initializes the minimum and current zoom levels."""
        scene_rect = self.sceneRect()
        self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)

        transform = self.transform()
        self._min_zoom = transform.m11()
        self._current_zoom = self._min_zoom

    def _adjust_view_to_scene(self) -> None:
        """Adjusts the view size and zoom so that the scene height fits 400 pixels,
        while preserving the scene's aspect ratio."""
        scene_rect = self.sceneRect()

        aspect_ratio = scene_rect.width() / scene_rect.height()
        desired_height = 400
        desired_width = int(aspect_ratio * desired_height)

        self.setFixedSize(desired_width, desired_height)

        self._min_zoom = 400/scene_rect.height()
        self._current_zoom = self._min_zoom

        self.resetTransform()
        transform_matrix = self.transform()
        transform_matrix.scale(self._min_zoom,self._min_zoom)
        self.setTransform(transform_matrix)