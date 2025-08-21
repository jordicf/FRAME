
from random import randint
from rectangles import RectObj
from graphical_view import GraphicalView
from PySide6.QtWidgets import QGraphicsItemGroup, QGraphicsItem, QGraphicsSceneMouseEvent, QGraphicsLineItem
from PySide6.QtCore import QPointF, Qt, QLineF
from PySide6.QtGui import QPen, QColor, QBrush, QPainterPath


N = "N"
W = "W"
S = "S"
E = "E"


class Module(QGraphicsItemGroup):
    """
    Groups a set of RectObj (rectangles) and manages them as a single item group.
    The module contains a main rectangle called the 'trunk', which is always present, and
    zero or more 'branches', which are additional rectangles connected to the trunk.
    
    Attribute:
    - soft: movable and resizable
    - hard: movable with fixed size
    - fixed: not movable and fixed size
    - terminal: 

    The group handles mouse events to enable interaction based on its attribute.
    Modules may also be connected to 'nets'.
    """

    name: str
    attribute: str
    trunk: RectObj # main rectangle
    branches: set[RectObj] # adjacent to the main rect
    grouped: bool = True
    nets: set['Net']

    def __init__(self, name: str, atr: str, trunk: RectObj, color: QColor|None = None) -> None:
        super().__init__()
        self.name = name
        self.attribute = atr
        self.trunk = trunk
        self.branches = set[RectObj]()
        self.nets = set[Net]()
        self.area = 0 # It will be updated

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

        if color:
            self.trunk.setBrush(color)
        else:
            self.trunk.setBrush(QColor(randint(10, 240), randint(10, 240), randint(10, 240)))
        
        self.add_rect_to_module(self.trunk, is_trunk=True)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        Handles mouse press on the module.
        Clears selection of all other items and selects this module exclusively. If the module 
        is not 'fixed', changes the cursor to an open hand.
        """

        self.focusItem()
        for item in self.scene().selectedItems():
            item.setSelected(False)

        self.setSelected(True)
        if self.attribute != "fixed":
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Updates the position of the nets when the mouse is released, to ensure
        that they end in the correct position after fast movements."""
        self.update_nets()
        return super().mouseReleaseEvent(event)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: QPointF) -> QPointF:
        """Keeps the module inside the scene rectangle on position change, and updates the nets accordingly."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            new_pos = value
            scene_rect = self.scene().sceneRect()

            # boundingRect() in local coordinates -> map it to scene coordinates
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
                self.update_nets()
                return QPointF(new_pos.x() + dx, new_pos.y() + dy)
            
        self.update_nets()
        return super().itemChange(change, value)
    
    def add_rect_to_module(self, rect: RectObj, is_trunk: bool = False) -> None:
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
            pen.setWidth(1)
            rect.setPen(pen)

        if not is_trunk:
            rect.setBrush(self.trunk.brush()) # same color
            self.branches.add(rect)
        self.addToGroup(rect)
        self.area += rect.area
    
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
        """
        Calculates the direction to position the given rectangle relative to the trunk.
        Returns one of 'N', 'W', 'S' or 'W' indicating where the rectanfke should be located
        respect to the trunk.
        """
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
        """Aligns the branches around the trunk according to their initial location."""

        for branch in self.branches:
            trunk_midp = self.trunk.midpoint()
            trunk_w, trunk_h = self.trunk.rect().width(), self.trunk.rect().height()
            trunk_x, trunk_y = trunk_midp.x(), trunk_midp.y()
        
            loc = self._location(branch)

            branch_midp = branch.midpoint()
            branch_w, branch_h = branch.rect().width(), branch.rect().height()

            new_y = branch_midp.y()
            if loc in "NS": # Swap x and y coordinates and widths/heights to reuse the logic designed for E/W alignment
                trunk_x, trunk_y = trunk_y, trunk_x
                trunk_w, trunk_h = trunk_h, trunk_w

                branch_w, branch_h = branch_h, branch_w
                new_y = branch_midp.x()

            if loc in "ES":
                new_x = trunk_x + (trunk_w + branch_w)/2
            else:
                new_x = trunk_x - (trunk_w + branch_w)/2
            
            max_y, min_y = trunk_y + (trunk_h - branch_h)/2, trunk_y + (branch_h - trunk_h)/2
            if branch_h > trunk_h: # If the branch is bigger than the trunk
                new_y = max_y if abs(new_y-max_y) < abs(new_y-min_y) else min_y
            else:
                new_y = max(min(new_y, max_y), min_y)
            
            if loc in "EW":
                branch.setPos(branch.midpoint_to_topleft(QPointF(new_x,new_y)))
            else:
                branch.setPos(branch.midpoint_to_topleft(QPointF(new_y,new_x)))

    def regroup(self) -> None:
        """Adds the trunk and branches back to the group and joins them together. 
        Also updates nets' position and the area, in case any rectangles were resized while ungrouped."""
        self.join()
        self.scene().clearSelection()
        self.addToGroup(self.trunk)
        for rect in self.branches:
            self.addToGroup(rect)
        
        self.grouped = True
        
        self.update_nets()
        self.update_area()

    def ungroup(self) -> None:
        """Removes the trunk and branches from the group, but keeps them as attributes of the class."""
        self.removeFromGroup(self.trunk)
        for rect in self.branches:
            self.removeFromGroup(rect)
        
        self.grouped = False

    def update_area(self) -> None:
        """Recalculates and updates the total area of the module."""
        area = self.trunk.area
        for branch in self.branches:
            area += branch.area

        self.area = area

    def centroid(self) -> QPointF:
        """Calculates and returns the centroid of the module."""
        total_area = 0
        x, y = 0, 0

        for rectobj in {self.trunk} | self.branches:
            center = rectobj.midpoint()
            rect_area = rectobj.area

            x += rect_area * center.x()
            y += rect_area * center.y()
            total_area += rect_area

        return QPointF(x/total_area, y/total_area)

    def add_net(self, net: 'Net') -> None:
        """Adds a given net to the set if nets connected to this module."""
        self.nets.add(net)

    def update_nets(self) -> None:
        """Updates all the nets connected to the module."""
        for net in self.nets:
            net.update_net()

    def connected_modules(self) -> list[str]:
        """Returns a sorted list with the names of the modules it's connected to."""
        connected_mod = {mod.name for net in self.nets for mod in net.modules}
        connected_mod.discard(self.name)
        return sorted(connected_mod)

    def shape(self) -> QPainterPath:
        """Returns the selectable shape of the module. It only makes selectable the area with
        the rectangles, not the full bounding rect."""

        path = QPainterPath()
        path.addPath(self.trunk.shape().translated(self.trunk.pos()))
        for rect in self.branches:
            path.addPath(rect.shape().translated(rect.pos()))
        return path


class Net:
    """
    A class to represent a net connecting multiple modules in a graphical view. The connections
    are visualized as lines drawn from the centroid of the net to the centroids of each connected
    module.
    """
    _view: GraphicalView # The view where the net is shown
    _modules: set[Module] # The modules this net connects
    _lines: set[QGraphicsLineItem]
    _weight: float
    _visible: bool = True

    def __init__(self, view: GraphicalView, modules: set[Module], weight: float) -> None:
        self._view = view
        self._modules = modules
        self._lines = set[QGraphicsLineItem]()
        self._weight = weight

        for module in self._modules:
            module.add_net(self)

        self.update_net()

    def update_net(self) -> None:
        """Updates the net by redrawing lines between modules or to the centroid."""
        self.clear_lines()

        if len(self._modules) == 2: # Just a single line connecting the 2 modules
            modules = self._modules.copy()
            line = QGraphicsLineItem(QLineF(modules.pop().centroid(), modules.pop().centroid()))
            self._register_line(line)
        else: # Hyperedge
            centroid = self.centroid_modules()
            for module in self._modules:
                line = QGraphicsLineItem(QLineF(centroid, module.centroid()))
                self._register_line(line)

    def _register_line(self, line: QGraphicsLineItem) -> None:
        """Adds the given line to the internal set and shows it in the view if currently visible."""
        line.setOpacity(0.5)
        self._lines.add(line)
        self._view.show_item(line)
        if not self._visible:
            line.hide()

    def centroid_modules(self) -> QPointF:
        """Calculates the centroid of the modules this net connects."""
        total_area = 0
        x, y = 0, 0

        for mod in self._modules:
            cent = mod.centroid()
            mod_area = mod.area

            x += mod_area * cent.x()
            y += mod_area * cent.y()
            total_area += mod_area

        return QPointF(x/total_area, y/total_area)

    def hide(self) -> None:
        """Hides all lines of the net."""
        for line in self._lines:
            line.setVisible(False)
        self._visible = False

    def show(self) -> None:
        """Makes all lines of the net visible."""
        for line in self._lines:
            line.setVisible(True)
        self._visible = True

    def clear_lines(self) -> None:
        """Removes and clears all the lines from the net."""
        for line in self._lines:
            line.scene().removeItem(line)

        self._lines.clear()

    @property
    def modules(self) -> set[Module]:
        "Returns the set of modules the net connects."
        return self._modules