from __future__ import annotations
from PySide6.QtWidgets import (
    QGraphicsRectItem,
    QGraphicsItem,
    QGraphicsSceneMouseEvent,
    QGraphicsSceneHoverEvent
)
from PySide6.QtCore import Qt, QRectF, QPointF

class RectObj(QGraphicsRectItem):
    """
    A class that provides a customized rectangle item that you can add to a QGraphicsScene.
    Optionally includes handles for interactive resizing.
    """

    _handles: dict[str,Handle] # key: corner, value: Handle
    _area: float

    def __init__(self, x: float, y: float, w: float, h: float):
        super().__init__(0, 0, w, h)
        self.setPos(x, y)
        self._handles = dict[str,Handle]()
        self._area = w * h

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setOpacity(0.4)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Sets the cursor to an open hand when the rectangle is pressed.
        Clears selection of all other items and selects this rectangle exclusively."""
        self.focusItem()
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        for item in self.scene().selectedItems():
            item.setSelected(False)

        self.setSelected(True)

    def itemChange(self, change: QGraphicsRectItem.GraphicsItemChange, value: QPointF):
        """Adjusts the rectangle's position to prevent it from moving outside the scene."""
        if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionChange and self.scene():
            new_pos = value
            scene_rect = self.scene().sceneRect()
            rect = self.rect()
            if not scene_rect.contains(QRectF(new_pos, rect.size())):
                new_pos.setX(min(scene_rect.right() - rect.width() - 1, max(new_pos.x(), scene_rect.left())))
                new_pos.setY(min(scene_rect.bottom() - rect.height() - 1, max(new_pos.y(), scene_rect.top())))
                return new_pos
        return super().itemChange(change, value)
    
    def copy(self) -> 'RectObj':
        """Returns a copy of the rectangle."""        
        rect = RectObj(self.scenePos().x(), self.scenePos().y(), self.rect().width(), self.rect().height())
        rect.setBrush(self.brush())
        return rect
    
    def midpoint(self) -> QPointF:
        """Returns de center point of the rectangle (in scene coordinates)."""
        point, rect = self.scenePos(), self.rect()

        x = point.x() + rect.width()/2
        y = point.y() + rect.height()/2
        return QPointF(x,y)
    
    def midpoint_to_topleft(self, mid: QPointF) -> QPointF:
        """Given the center point of the rectangle, returns its top-left corner in scene
        coordinates."""
        rect = self.rect()
        x = mid.x() - rect.width()/2
        y = mid.y() - rect.height()/2
        return QPointF(x,y)
    
    def create_handles(self) -> None:
        """Creates 4 Handles for the rectangle and positions them acordingly."""
        for corner in ("top-left", "top-right", "bottom-right", "bottom-left"):
            handle = Handle(corner, self)
            self._handles[corner] = handle
        self.update_handles_position()

    def update_handles_position(self) -> None:
        """Positions the handles at the corners of the rectangle."""
        rect = self.rect()
        
        self._handles["top-left"].setPos(rect.topLeft())
        self._handles["top-right"].setPos(rect.topRight())
        self._handles["bottom-right"].setPos(rect.bottomRight())
        self._handles["bottom-left"].setPos(rect.bottomLeft())

    def resize_from_handle(self, corner: str, scene_pos: QPointF) -> None:
        """Resizes the rectangle based on the new position of the specified corner's handle."""
        local_pos = self.mapFromScene(scene_pos)
        new_rect = self.rect()

        if corner == "top-left":
            new_rect.setTopLeft(local_pos)
        elif corner == "top-right":
            new_rect.setTopRight(local_pos)
        elif corner == "bottom-right":
            new_rect.setBottomRight(local_pos)
        elif corner == "bottom-left":
            new_rect.setBottomLeft(local_pos)

        # Prevent creating a rectangle that is too small
        if new_rect.width() < 10 or new_rect.height() < 10:
            return

        self.prepareGeometryChange()
        self.setRect(new_rect)
        self.update_handles_position()

    def reset_local_origin_update_area(self) -> None:
        """Updates local coordinates so that (0,0) corresponds to the rectangle's top-left
        corner. It also updates the area of the rectangle."""
        old_rect = self.rect()
        top_left_scene = self.mapToScene(old_rect.topLeft()) # top-left corner of the rect in scene coorfinates

        self.prepareGeometryChange()
        self.setPos(top_left_scene)
        w, h = old_rect.width(), old_rect.height()
        self.setRect(0, 0, w, h)
        self._area = w * h
    
    @property
    def area(self) -> float:
        """Returns the area of the rectangle."""
        return self._area

    
HANDLE_SIZE = 10

class Handle(QGraphicsRectItem):
    """
    A class to create a handle used to resize a parent RectObj.
    This handle appears as a semi-transparent square positioned at the specified
    corner of the parent item.
    """
    
    corner: str

    def __init__(self, corner: str, parent: RectObj):   
        """Initializes a new Handle instance for resizing the given parent RectObj."""
        if corner == "top-left":
            x, y = 0, 0
        elif corner == "top-right":
            x, y = -HANDLE_SIZE, 0
        elif corner == "bottom-right":
            x, y = -HANDLE_SIZE, -HANDLE_SIZE
        elif corner == "bottom-left":
            x, y = 0, -HANDLE_SIZE
        else:
            assert False, "Incorrect corner"
        super().__init__(x, y, HANDLE_SIZE, HANDLE_SIZE)
        self.corner = corner
        self.setParentItem(parent)
        self.setAcceptHoverEvents(True)

        self.setBrush(Qt.GlobalColor.transparent)
        self.setOpacity(0.1)
        self.setFlags(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable | QGraphicsRectItem.GraphicsItemFlag.ItemSendsScenePositionChanges)

    def itemChange(self, change: QGraphicsRectItem.GraphicsItemChange, value: QPointF):
        """Handles item position changes and resizes the parent RectObj accordingly."""
        if change == QGraphicsRectItem.GraphicsItemChange.ItemScenePositionHasChanged:
            parent = self.parentItem()
            assert isinstance(parent, RectObj)
            parent.resize_from_handle(self.corner, value)
        return super().itemChange(change, value)
    
    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Updates the handle's position to ensure it remains correctly aligned with the resized parent and
        updates the local coordinates of the parent."""
        parent = self.parentItem()
        assert isinstance(parent, RectObj)
        parent.reset_local_origin_update_area()
        parent.update_handles_position()

        return super().mouseReleaseEvent(event)
    
    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent):
        """Changes the cursor to a diagonal resize icon when the mouse hovers over the handle."""
        if self.corner in ("top-left", "bottom-right"):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent):
        """Restores the default cursor when the mouse leaves the handle area."""
        self.unsetCursor()
        super().hoverLeaveEvent(event)