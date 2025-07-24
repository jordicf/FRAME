from __future__ import annotations
from PySide6.QtWidgets import (
    QGraphicsRectItem,
    QGraphicsItem,
    QGraphicsSceneMouseEvent,
    QGraphicsSceneHoverEvent
)
from PySide6.QtCore import Qt, QRectF, QPointF


class RectObj(QGraphicsRectItem):
    """A class that provides a customized rectangle item that you can add to a QGraphicsScene.
    Optionally includes handles for interactive resizing."""

    handles: dict[str,Handle]

    def __init__(self, x: float, y: float, w: float, h: float):
        super().__init__(0, 0, w, h)
        self.setPos(x, y)
        self.handles = dict[str,Handle]()

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setOpacity(0.4)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Sets the cursor to an open hand when the rectangle is pressed."""
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
        """Returns de center point of the rectangle."""
        point, rect = self.scenePos(), self.rect()

        x = point.x() + rect.width()/2
        y = point.y() + rect.height()/2
        return QPointF(x,y)
    
    def midpoint_to_topleft(self, mid: QPointF) -> QPointF:
        """Given the center point of the rectangle, returns the top-left point of the rectangle."""
        rect = self.rect()
        x = mid.x() - rect.width()/2
        y = mid.y() - rect.height()/2
        return QPointF(x,y)
    
    def create_handles(self) -> None:
        """Creates 4 Handles for the rectangle and positions them acordingly."""
        for corner in ("top-left", "top-right", "bottom-right", "bottom-left"):
            handle = Handle(corner, self)
            self.handles[corner] = handle
        self.update_handle_position()

    def update_handle_position(self) -> None:
        """Positions the handles at the corners of the rectangle."""
        rect = self.rect()
        
        self.handles["top-left"].setPos(rect.topLeft())
        self.handles["top-right"].setPos(rect.topRight())
        self.handles["bottom-right"].setPos(rect.bottomRight())
        self.handles["bottom-left"].setPos(rect.bottomLeft())

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
        self.update_handle_position()


    def reset_local_origin(self):
        """Updates local coordinates so that (0,0) corresponds to the rectangle's top-left
        corner."""
        old_rect = self.rect()
        top_left_scene = self.mapToScene(old_rect.topLeft())

        self.prepareGeometryChange()
        self.setPos(top_left_scene)
        self.setRect(0, 0, old_rect.width(), old_rect.height())

    def info(self) -> None:
        print("rect topLeft", self.rect().topLeft())
        print("top-left mapped to scene : ", self.mapToScene(self.rect().topLeft()))
        print()
        print("width", self.rect().width())
        print("height", self.rect().height())
        print()
        print("pos", self.pos())
        print("scenepos", self.scenePos())

        print("-" * 10)
        print()

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.info()
        return super().mouseDoubleClickEvent(event)
    
HANDLE_SIZE = 10

class Handle(QGraphicsRectItem):
    """A class to create a handle used to resize a parent RectObj.
    This handle appears as a semi-transparent square positioned at the specified
    corner of the parent item."""
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
        parent.reset_local_origin()
        parent.update_handle_position()

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
