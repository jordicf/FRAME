
from random import randint
from rectangles import RectObj
from PySide6.QtWidgets import QGraphicsItemGroup, QGraphicsItem, QGraphicsSceneMouseEvent
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPen, QColor, QBrush, QPainterPath

N = "N"
W = "W"
S = "S"
E = "E"


class Module(QGraphicsItemGroup):
    """
    Groups a set of RectObj (rectangles) and manages them as a single item group.

    Attribute:
    - soft: movable and resizable
    - hard: movable with fixed size
    - fixed: not movable and fixed size
    - terminal: 

    The group handles mouse events to enable interaction based on its attribute.
    """

    name: str
    attribute: str
    trunk: RectObj # main rectangle
    branches: set[RectObj] # adjacent to the main rect
    grouped: bool = True

    def __init__(self, name: str, atr: str, rect: RectObj, color: QColor|None = None) -> None:
        super().__init__()
        self.name = name
        self.attribute = atr
        self.trunk = rect
        self.branches = set[RectObj]()

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

        if color:
            self.trunk.setBrush(color)
        else:
            self.trunk.setBrush(QColor(randint(10, 240), randint(10, 240), randint(10, 240)))
        
        self.add_rect_to_module(self.trunk, True)


    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.focusItem()
        for item in self.scene().selectedItems():
            item.setSelected(False)

        self.setSelected(True)
        if self.attribute != "fixed":
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: QPointF) -> QPointF:
        

        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            new_pos = value
            scene_rect = self.scene().sceneRect()

            # boundingRect() in local coordinates -> map it ot the coordinates of the scene
            rect = self.mapRectToScene(self.boundingRect())

            moved_rect = rect.translated(new_pos - self.pos()) # move to the new position

            dx = dy = 0.0

            if moved_rect.left() < scene_rect.left():
                dx = scene_rect.left() - moved_rect.left()
            elif moved_rect.right() > scene_rect.right():
                dx = scene_rect.right() - moved_rect.right()

            if moved_rect.top() < scene_rect.top():
                dy = scene_rect.top() - moved_rect.top()
            elif moved_rect.bottom() > scene_rect.bottom():
                dy = scene_rect.bottom() - moved_rect.bottom()

            if dx or dy:
                return QPointF(new_pos.x() + dx, new_pos.y() + dy)
            
        # if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
        #     if self.isSelected():
        #         print()

        return super().itemChange(change, value)
    
    def add_rect_to_module(self, rect: RectObj, trunk: bool = False) -> None:
        """Adds the given rectangle to the module and sets its properties based on its role 
        (trunk or branch) and the module's attribute."""
        rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

        if self.attribute == "hard" or self.attribute == "soft":
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
            if self.attribute == "soft":
                self.setAcceptHoverEvents(True)
                rect.create_handles()
        elif self.attribute == "fixed":
            rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            pen = QPen()
            pen.setColor(Qt.GlobalColor.red)
            pen.setWidth(2)
            rect.setPen(pen)

        if not trunk:
            rect.setBrush(self.trunk.brush()) # same color
            self.branches.add(rect)
        self.addToGroup(rect)
    
    # def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
    #     if self.grouped:
    #         for rect in self.branches:
    #             self.removeFromGroup(rect)
    #         self.grouped = False
    #     else:
    #         self.join()
    #         for rect in self.branches:
    #             self.addToGroup(rect)
    #         self.grouped = True

    #     return super().mouseDoubleClickEvent(event)
    
    def copy(self) -> 'Module':
        """Returns a copy of the module."""
        new_module = Module(self.name, self.attribute, self.trunk.copy(), self.trunk.brush().color())
        for rect in self.branches:
            new_module.add_rect_to_module(rect.copy())

        return new_module
    
    def setBrush(self, brush: QBrush) -> None:
        """Applies the given brush to the trunk and all rectangles in the module."""
        self.trunk.setBrush(brush)
        for rect in self.branches:
            rect.setBrush(brush)

    def _location(self, rectangle: RectObj) -> str:
        """Calculates the direction to position the given rectangle relative to the trunk."""
        trunk_midp = self.trunk.midpoint()
        trunk_rect = self.trunk.rect()

        rect_midp = rectangle.midpoint()
        rect = rectangle.rect()
        x_dist = abs(rect_midp.x() - trunk_midp.x()) - trunk_rect.width()/2 - rect.width()/2
        y_dist = abs(rect_midp.y() - trunk_midp.y()) - trunk_rect.height()/2 - rect.height()/2

        if x_dist > y_dist:
            if trunk_midp.x() < rect_midp.x():
                return E
            else:
                return W
        else:
            if trunk_midp.y() < rect_midp.y():
                return S
            else:
                return N
            
    def join(self) -> None:
        """Aligns the rectangles around the trunk according to their initial location."""
        for rect in self.branches:
            trunk_midp = self.trunk.midpoint()
            trunk_w, trunk_h = self.trunk.rect().width(), self.trunk.rect().height()
            trunk_x, trunk_y = trunk_midp.x(), trunk_midp.y()
        
            loc = self._location(rect)

            rect_midp = rect.midpoint()
            rect_w, rect_h = rect.rect().width(), rect.rect().height()

            new_y = rect_midp.y()
            if loc in "NS": # Swap x and y coordinates and widths/heights to reuse the logic designed for E/W alignment.
                trunk_x, trunk_y = trunk_y, trunk_x
                trunk_w, trunk_h = trunk_h, trunk_w

                rect_w, rect_h = rect_h, rect_w

                new_y = rect_midp.x()

            if loc in "ES":
                new_x = trunk_x + (trunk_w + rect_w)/2
            else:
                new_x = trunk_x - (trunk_w + rect_w)/2
            
            new_y = min(new_y, trunk_y + (trunk_h - rect_h)/2)
            new_y = max(new_y, trunk_y + (rect_h - trunk_h)/2)
            
            if loc in "EW":
                rect.setPos(rect.midpoint_to_topleft(QPointF(new_x,new_y)))
            else:
                rect.setPos(rect.midpoint_to_topleft(QPointF(new_y,new_x)))

    def regroup(self) -> None:
        """Adds the trunk and branches back to the group and joins them together."""

        self.join()
        self.scene().clearSelection()
        self.addToGroup(self.trunk)
        for rect in self.branches:
            self.addToGroup(rect)
        
        self.grouped = True

    def ungroup(self) -> None:
        """Removes the trunk and branches from the group, but keeps them as attributes of the class."""

        self.removeFromGroup(self.trunk)
        for rect in self.branches:
            self.removeFromGroup(rect)
        
        self.grouped = False

    def shape(self) -> QPainterPath:
        """Returns the selectable shape of the module. It only makes selectable the area with
        the rectangles, not the full bounding rect."""

        path = QPainterPath()
        path.addPath(self.trunk.shape().translated(self.trunk.pos()))
        for rect in self.branches:
            path.addPath(rect.shape().translated(rect.pos()))
        return path
    
    # def paint(self, painter: QPainter, option, widget=None):
    #     super().paint(painter, option, widget)

    #     if self.isSelected():
    #         pen = QPen(Qt.PenStyle.DotLine)
    #         pen.setColor(Qt.GlobalColor.black)
    #         pen.setWidth(2)
    #         painter.setPen(pen)
    #         painter.setBrush(Qt.BrushStyle.NoBrush)
            
    #         # Dibujamos la forma real del grupo
    #         path = QPainterPath()
    #         path.addPath(self.trunk.shape().translated(self.trunk.pos()))
    #         for rect in self.branches:
    #             path.addPath(rect.shape().translated(rect.pos()))

    #         painter.drawPath(path)
