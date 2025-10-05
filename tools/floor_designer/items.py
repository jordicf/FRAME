from __future__ import annotations
import rportion as rp # type: ignore
from random import randint
from graphical_view import GraphicalView
from PySide6.QtWidgets import (QGraphicsRectItem,QGraphicsItem,QGraphicsSceneMouseEvent,QGraphicsSceneHoverEvent,
    QStyleOptionGraphicsItem, QGraphicsItemGroup, QGraphicsLineItem, QWidget, QGraphicsTextItem
)
from PySide6.QtCore import QPointF, Qt, QLineF, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QFont

N = "N"
W = "W"
S = "S"
E = "E"

MIN_PIN_SIZE = 0.000000001 # Minimum pin size ti ensure it's visible
SCENE_MAGNET = 2

class Module(QGraphicsItemGroup):
    """
    Groups a set of RectObj (rectangles) and manages them as a single item group.
    The module contains a main rectangle called the 'trunk', which is always present, and
    zero or more 'branches', which are additional rectangles connected to the trunk.
    
    Attribute:
    - soft: movable and resizable
    - hard: movable with fixed size
    - fixed: not movable and fixed size

    The group handles mouse events to enable interaction based on its attribute.
    Modules may also be connected to 'fly lines'.

    I/O Pins can be represented using this same class, which makes it easier to integrate
    them into the application and connect them to fly lines. Existing functions can 
    handle them as modules with a few variations.
    """

    _name: str
    _attribute: str
    _trunk: RectObj # main rectangle
    _branches: set[RectObj] # adjacent to the main rect
    _grouped: bool = True
    _fly_lines: set['FlyLine']
    _area: float
    _maintain_area: bool = False # used for resizing soft modules
    _is_iopin: bool
    _blockages: set[QGraphicsRectItem] # blockages of the die

    def __init__(self, name: str, atr: str, trunk: RectObj, iopin: bool = False, color: QColor|None = None) -> None:
        super().__init__()
        self._name = name
        self._attribute = atr
        self._trunk = trunk
        self._branches = set[RectObj]()
        self._fly_lines = set[FlyLine]()
        self._area = 0 # It will be updated
        self._is_iopin = iopin
        self._blockages = set[QGraphicsRectItem]()

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

        if color:
            self._trunk.setBrush(color)
        else:
            self._trunk.setBrush(QColor(randint(10, 240), randint(10, 240), randint(10, 240)))
        
        self.add_rect_to_module(self._trunk, is_trunk=True)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        Handles mouse press on the module.
        Clears selection of all other items and selects this module exclusively. If the module 
        is not 'fixed', changes the cursor to an open hand.
        """
        if self.is_iopin or self.shape().contains(event.pos()):
            self.focusItem()
            for item in self.scene().selectedItems():
                item.setSelected(False)

            self.setSelected(True)
            if self._attribute != "fixed":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Updates the position of the fly lines when the mouse is released, to ensure
        that they end in the correct position after fast movements."""
        self.update_fly_lines()
        return super().mouseReleaseEvent(event)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: QPointF) -> QPointF:
        """Keeps the module aligned with the scene rectangle on position change, and updates the fly lines
        accordingly. I/O pins are also aligned with the blockages."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            new_pos = value
            dx = dy = 0.0
            rect = self.mapRectToScene(self.boundingRect())
            moved_rect = rect.translated(new_pos - self.pos()) # move to the new position

            magnet = SCENE_MAGNET * self.transform().m11()

            scene_rect = self.scene().sceneRect()
            alignment_rects = [scene_rect]
            if self._is_iopin:
                alignment_rects = alignment_rects + [block.mapRectToScene(block.rect()) for block in self._blockages]

            for alignment_rect in alignment_rects:

                # Horizontal alignment
                if abs(moved_rect.left() - alignment_rect.left()) < magnet:
                    dx = alignment_rect.left() - moved_rect.left()
                elif abs(moved_rect.right() - alignment_rect.right()) < magnet:
                    dx = alignment_rect.right() - moved_rect.right()
                elif abs(moved_rect.left() - alignment_rect.right()) < magnet:
                    dx = alignment_rect.right() - moved_rect.left()
                elif abs(moved_rect.right() - alignment_rect.left()) < magnet:
                    dx = alignment_rect.left() - moved_rect.right()

                # Vertical alignment
                if abs(moved_rect.top() - alignment_rect.top()) < magnet:
                    dy = alignment_rect.top() - moved_rect.top()
                elif abs(moved_rect.bottom() - alignment_rect.bottom()) < magnet:
                    dy = alignment_rect.bottom() - moved_rect.bottom()
                elif abs(moved_rect.top() - alignment_rect.bottom()) < magnet:
                    dy = alignment_rect.bottom() - moved_rect.top()
                elif abs(moved_rect.bottom() - alignment_rect.top()) < magnet:
                    dy = alignment_rect.top() - moved_rect.bottom()

                if dx or dy:
                    self.update_fly_lines()
            
            return QPointF(new_pos.x() + dx, new_pos.y() + dy)
            
        self.update_fly_lines()
        return super().itemChange(change, value)
    
    def add_rect_to_module(self, rect: RectObj, is_trunk: bool = False) -> None:
        """Adds the given rectangle to the module and sets its properties based on its role 
        (trunk or branch), the module's attribute, and if it is an I/O pin."""
        rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        rect.maintain_area = self.maintain_area
        rect.set_module(self)
        self.addToGroup(rect)

        if self._is_iopin:
            rect.is_iopin = True
            pen = QPen()
            pen.setColor(self._trunk.brush().color())
            pen.setCosmetic(True)
            pen.setWidth(10)
            rect.setPen(pen)
            rect.setOpacity(1)

        if self._attribute == "hard" or self._attribute == "soft":
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
            rect.set_orientation()
            if self._attribute == "soft":
                self.setAcceptHoverEvents(True)
                rect.create_handles()
        elif self._attribute == "fixed":
            rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            if not self._is_iopin:
                pen = QPen()
                pen.setColor(Qt.GlobalColor.red)
                pen.setWidth(3)
                pen.setCosmetic(True)
                rect.setPen(pen)

        if is_trunk:
            rect.set_as_trunk()
        else:
            rect.setBrush(self._trunk.brush()) # same color
            self._branches.add(rect)

        self._area += rect.area
    

    def copy(self) -> 'Module':
        """Returns a copy of the module."""
        new_module = Module(self._name, self._attribute, self._trunk.copy(), self.is_iopin, color=self._trunk.brush().color())
        for rect in self._branches:
            new_module.add_rect_to_module(rect.copy())

        return new_module
    
    def setBrush(self, brush: QBrush) -> None:
        """Applies the given brush to the trunk and all rectangles in the module."""
        self._trunk.setBrush(brush)
        for rect in self._branches:
            rect.setBrush(brush)

    def setPen(self, pen: QPen) -> None:
        """Applies the given pen to the trunk and all rectangles in the module."""
        self._trunk.setPen(pen)
        for rect in self._branches:
            rect.setPen(pen)


    def join(self) -> None:
        """Aligns the branches around the trunk according to their initial location. After all branches are placed,
        if any two overlap, cuts one of the two branches' rectangle to remove the overlapping part."""
        
        locations = dict[str,list[RectObj]]() # key: trunk side (N,S,W,E), value: list of branches on that side

        for branch in self._branches:
            # Trunk dimensions and center (midpoint)
            trunk_midp = self._trunk.midpoint()
            trunk_w, trunk_h = self._trunk.rect().width(), self._trunk.rect().height()
            trunk_x, trunk_y = trunk_midp.x(), trunk_midp.y()

            # Branch dimensions and midpoint
            branch_midp = branch.midpoint()
            branch_w, branch_h = branch.rect().width(), branch.rect().height()

            # Identify which side of the trunk the branch belongs to
            loc = self._location(branch)
            if locations.get(loc, None) is None:
                locations[loc] = [branch]
            else:
                locations[loc].append(branch)

            new_y = branch_midp.y()

            if loc in "NS": # Swap x and y coordinates and widths/heights to reuse the logic designed for E/W alignment
                trunk_x, trunk_y = trunk_y, trunk_x
                trunk_w, trunk_h = trunk_h, trunk_w

                branch_w, branch_h = branch_h, branch_w
                new_y = branch_midp.x()

            if loc in "ES":
                new_x = trunk_x + (trunk_w + branch_w)/2
            else: # N,W
                new_x = trunk_x - (trunk_w + branch_w)/2
            
            # Vertical alignment limits relative to the trunk
            max_y, min_y = trunk_y + (trunk_h - branch_h)/2, trunk_y + (branch_h - trunk_h)/2

            # If branch is taller than trunk, align it to nearest corner; otherwise fit inside the limit
            if branch_h > trunk_h or self._is_iopin:
                new_y = max_y if abs(new_y-max_y) < abs(new_y-min_y) else min_y
            else:
                new_y = max(min(new_y, max_y), min_y)
            
            if loc in "EW":
                branch.setPos(branch.midpoint_to_topleft(QPointF(new_x,new_y)))
            else:
                branch.setPos(branch.midpoint_to_topleft(QPointF(new_y,new_x)))

        self._legalize_branches(locations)

    def _location(self, rectangle: RectObj) -> str:
        """
        Calculates the direction to position the given rectangle relative to the trunk.
        Returns one of 'N', 'W', 'S' or 'W' indicating where the rectanfke should be located
        respect to the trunk.
        """
        trunk_midp = self._trunk.midpoint()
        trunk_rect = self._trunk.rect()

        rect_midp = rectangle.midpoint()
        rect = rectangle.rect()
        x_dist = abs(rect_midp.x() - trunk_midp.x()) - trunk_rect.width()/2 - rect.width()/2
        y_dist = abs(rect_midp.y() - trunk_midp.y()) - trunk_rect.height()/2 - rect.height()/2

        if (not rectangle.is_iopin and x_dist > y_dist) or (rectangle.is_iopin and not rectangle.is_horizontal):
            if trunk_midp.x() < rect_midp.x():
                return E
            else:
                return W
        else:
            if trunk_midp.y() < rect_midp.y():
                return S
            else:
                return N

    def _legalize_branches(self, locations: dict[str,list[RectObj]]) -> None:
        """Given a dictionary that indicates the side where the branches are located, checks overlaps
        between branches on the same side and cuts the overlapping are from one of them."""

        for loc in locations.keys():
            # Sort branches of this side by size (largest first) to preserve the biggest
            if loc in "WE":
                sorted_side = sorted(locations[loc], key= lambda branch: -branch.rect().width())
            else: # N,S
                sorted_side = sorted(locations[loc], key= lambda branch: -branch.rect().height())

            # Compare each branch with the ones already placed (the biggest one is never cut)
            for i, rect in enumerate(sorted_side):
                for rect_placed in sorted_side[:i]:
                    # if they overlap, cut th overlapping part of the current branch
                    if rect.collidesWithItem(rect_placed): 
                        if loc in "WE":
                            self._cut_item_e_w(rect_placed, rect)
                        else:
                            self._cut_item_n_s(rect_placed, rect)
    
    def _cut_item_e_w(self, fixed: RectObj, rect: RectObj) -> None:
        """
        This function is only for rectangles located to the east or west of the trunk.

        Cuts from 'rect' the part that overlaps with 'fixed'. 
        - If the overlap is in the middle of 'rect' (meaning 'rect' extends both above and below the overlap), 
        a new rectangle is created for the upper part.
        - If 'fixed' fully contains 'rect', then 'rect' is deleted.
        
        Note: in PySide6 the Y-axis origin (0,0) is at the top-left corner and increases downwards
        (opposite of the usual Cartesian system).
        """

        fixed_y, rect_y = fixed.midpoint().y(), rect.midpoint().y()
        fixed_h, rect_h = fixed.rect().height(), rect.rect().height()
        rect_top, rect_botm = rect_y - rect_h/2, rect_y + rect_h/2

        # y-coordinate range where they overlap
        top = max(fixed_y - fixed_h/2, rect_top)
        bottom = min(fixed_y + fixed_h/2, rect_botm)

        if top > rect_top or bottom < rect_botm:
            new_rect = rect.rect()
            top_mapped = rect.mapFromScene(QPointF(0,top)).y()
            bottom_mapped = rect.mapFromScene(QPointF(0,bottom)).y()

            if top > rect_top: # exceeds above
                if bottom < rect_botm: # exceeds both above and below
                    # create a new rectangle for the upper part (the lower part will be cut later)
                    top_left = rect.scenePos()
                    w, h = new_rect.width(), top-top_left.y()
                    rect2 = RectObj(top_left.x() + w/2, top_left.y() + h/2, w, h)
                    self.add_rect_to_module(rect2)
                else: # only exceeds above
                    new_rect.setBottom(top_mapped)
            if bottom < rect_botm: # exceeds below
                new_rect.setTop(bottom_mapped)

            # Apply the updated rectangle
            rect.prepareGeometryChange()
            rect.setRect(new_rect)
            rect.reset_local_origin_update_area()
            rect.update_handles_position()
        elif top == rect_top and bottom == rect_botm: # 'fixed' fully contains 'rect'
            self._branches.discard(rect)
            self.scene().removeItem(rect)

    def _cut_item_n_s(self, fixed: RectObj, rect: RectObj) -> None:
        """
        This function is only for rectangles located to the north or south of the trunk.

        Cuts from 'rect' the part that overlaps with 'fixed'. 
        - If the overlap is in the middle of 'rect' (meaning 'rect' extends both left and right of the overlap), 
        a new rectangle is created for the left part.
        - If 'fixed' fully contains 'rect', then 'rect' is deleted.
        
        Note: in PySide6 the Y-axis origin (0,0) is at the top-left corner and increases downwards
        (opposite of the usual Cartesian system). The X-axis behaves as usual.
        """
        fixed_x, rect_x = fixed.midpoint().x(), rect.midpoint().x()
        fixed_w, rect_w = fixed.rect().width(), rect.rect().width()
        rect_left, rect_right = rect_x - rect_w/2, rect_x + rect_w/2

        # x-coordinate range where they overlap
        left = max(fixed_x - fixed_w/2, rect_left)
        right = min(fixed_x + fixed_w/2, rect_right)

        if left > rect_left or right < rect_right:
            new_rect = rect.rect()
            left_mapped = rect.mapFromScene(QPointF(left,0)).x()
            right_mapped = rect.mapFromScene(QPointF(right,0)).x()

            if left > rect_left: # exceeds left
                if right < rect_right: # exceeds both left and right
                    # create a new rectangle for the left part (the right part will be cut later)
                    top_left = rect.scenePos()
                    w, h = left - top_left.x(), new_rect.height()
                    rect2 = RectObj(top_left.x() + w/2, top_left.y() + h/2, w, h)
                    self.add_rect_to_module(rect2)
                else: # exceeds only left
                    new_rect.setRight(left_mapped)
            if right < rect_right: # exceeds right
                new_rect.setLeft(right_mapped)

            # Apply the updated rectangle
            rect.prepareGeometryChange()
            rect.setRect(new_rect)
            rect.reset_local_origin_update_area()
            rect.update_handles_position()
        elif left == rect_left and right == rect_right: # 'fixed' fully contains 'rect'
            self._branches.discard(rect)
            self.scene().removeItem(rect)

    def regroup(self) -> None:
        """Adds the trunk and branches back to the group and joins them together. 
        Also updates fly-lines' position and the area, in case any rectangles were resized while ungrouped."""
        if not self._grouped:
            self.join()
            self.scene().clearSelection()
            self.addToGroup(self._trunk)
            for rect in self._branches:
                self.addToGroup(rect)
            
            self._grouped = True
            
            self.update_fly_lines()
            self.update_area()

    def ungroup(self) -> None:
        """Removes the trunk and branches from the group, but keeps them as attributes of the class."""
        if self._grouped:
            self.removeFromGroup(self._trunk)
            for rect in self._branches:
                self.removeFromGroup(rect)
            
            self._grouped = False


    def update_area(self) -> None:
        """Recalculates and updates the total area of the module."""
        area = self._trunk.area
        for branch in self._branches:
            area += branch.area

        self._area = area

    def centroid(self) -> QPointF:
        """Calculates and returns the centroid of the module."""
        if self._is_iopin:
            return self._trunk.midpoint()

        total_area = 0.0
        x, y = 0.0, 0.0

        for rectobj in {self._trunk} | self._branches:
            center = rectobj.midpoint()
            rect_area = rectobj.area

            x += rect_area * center.x()
            y += rect_area * center.y()
            total_area += rect_area

        return QPointF(x/total_area, y/total_area)

    def add_fly_line(self, fly_line: 'FlyLine') -> None:
        """Adds a given fly line to the set of fly lines connected to this module."""
        self._fly_lines.add(fly_line)

    def update_fly_lines(self) -> None:
        """Updates all the fly lines connected to the module."""
        for fly_line in self._fly_lines:
            fly_line.update_fly_line()

    def connected_modules(self) -> list[str]:
        """Returns a sorted list with the names of the modules it's connected to."""
        connected_mod = {mod._name for fly_line in self._fly_lines for mod in fly_line.modules}
        connected_mod.discard(self._name)
        return sorted(connected_mod)

    def has_fly_line(self, fly_line: FlyLine) -> bool:
        """Returns if the given fly line belongs to this module."""
        return fly_line in self._fly_lines


    def shape(self) -> QPainterPath:
        """Returns the selectable shape of the module. It only makes selectable the area with
        the rectangles, not the full bounding rect."""

        path = QPainterPath()
        for r in self._branches | {self._trunk}:
            rect = r.mapRectToParent(r.rect())
            path.addRect(rect)

        return path

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, /, widget: QWidget | None = ...) -> None: # type: ignore
        """Draw the item, using a thicker border when it's selected."""
        if self.isSelected() and self._grouped:
            sel_pen = QPen()
            sel_pen.setWidth(3)
            sel_pen.setCosmetic(True)
            painter.setPen(sel_pen)
            painter.drawPath(self.shape())
        else:
            return super().paint(painter, option, widget)

    def boundingRect(self) -> QRectF:
        """Returns the bounding rectangle af all the rectangles of the module."""
        rect = QRectF()
        for r in self._branches | {self._trunk}:
            rect = rect.united(r.mapRectToParent(r.rect()))
        
        return rect

    def set_blockages(self, blockages: set[QGraphicsRectItem]) -> None:
        """Assigns the given set of blockage rectangles to the item, to be used later for alignment 
        during position changes."""
        self._blockages = blockages


    def adjust_pin_segments(self, segments: rp.RPolygon) -> None:
        """Adjusts pins made of 3 segments. Aligns the trunk with the nearest segment of the same
        orientation and resizes it to match its length. The branches are moved accordingly."""
        assert self._is_iopin and len(self.branches) == 2

        if self._grouped:
            self.ungroup()

        list_segments = list(segments.maximal_rectangles())
        trunk_center = self.trunk.midpoint()
        trunk_rect = self.trunk.rect()

        horizontal = self.trunk.is_horizontal
        for segment in list_segments:
            x_lower, x_upper = float(segment.x_lower), float(segment.x_upper) # type: ignore
            y_lower, y_upper = float(segment.y_lower), float(segment.y_upper) # type: ignore

            if horizontal:
                if y_lower == y_upper: # The segment is also horizontal
                    if abs(trunk_center.y() - y_lower) < MAGNET:
                        self._trunk.setPos(x_lower, y_lower)
                        self._trunk.prepareGeometryChange()
                        trunk_rect.setRight(self._trunk.mapFromScene(QPointF(x_upper,0)).x())
                        self._trunk.setRect(trunk_rect)
                        self._trunk.reset_local_origin_update_area()
                        self._trunk.update_handles_position()
            else:   
                if x_lower == x_upper: # The segment is vertical
                    if abs(trunk_center.x() - x_lower) < MAGNET:
                        self._trunk.setPos(x_lower, y_lower)
                        self.trunk.prepareGeometryChange()
                        trunk_rect.setBottom(self._trunk.mapFromScene(QPointF(0,y_upper)).y())
                        self._trunk.setRect(trunk_rect)
                        self._trunk.reset_local_origin_update_area()
                        self._trunk.update_handles_position()
  
        self.regroup()

        return None

    def legal_pin_location(self, segments: rp.RPolygon) -> bool:
        """Checks if the pin is placed in the given segments (in a legal location)."""

        assert self._is_iopin
        for item in {self.trunk} | self.branches:
            rect, center = item.rect(), item.midpoint()
            w, h = rect.width(), rect.height()
            
            if w == MIN_PIN_SIZE:
                w = 0
            if h == MIN_PIN_SIZE:
                h = 0

            segment = rp.rclosed(round(center.x()-w/2, 8), round(center.x()+w/2, 8), round(center.y()-h/2, 8), round(center.y()+h/2, 8)) # type: ignore
            if segments & segment != segment:
                return False
        
        return True

    def legal_module_location(self) -> bool:
        """Checks if the module is placed inside the scene and not intersecting any blockages."""
        scene_rect = self.scene().sceneRect()

        if not scene_rect.contains(self.mapRectToScene(self.boundingRect())):
            return False
        
        for rect in {self.trunk} | self.branches:
            rect_mapped = rect.mapRectToScene(rect.rect())

            for blockage in self._blockages:
                if rect_mapped.intersects(blockage.mapRectToScene(blockage.rect())):
                    return False
                
        return True

    @property
    def area(self):
        """Returns the total area of the module."""
        return self._area

    @property
    def maintain_area(self) -> bool:
        """Returns whether the area of the module's rectangles should be preserved when resizing."""
        return self._maintain_area

    @maintain_area.setter
    def maintain_area(self, maintain: bool) -> None:
        """Set whether the area should be preserved and apply this attribute to all of the rectangles."""
        self._maintain_area = maintain
        
        for rect in {self._trunk} | self._branches:
            rect.maintain_area = maintain
    
    @property
    def name(self) -> str:
        """The name of the module."""
        return self._name
    
    @property
    def attribute(self) -> str:
        """The attribute of the module (fixed, hard or soft)."""
        return self._attribute

    @property
    def trunk(self) -> RectObj:
        """The trunk (main rectangle) of the module."""
        return self._trunk

    @property
    def branches(self) -> set[RectObj]:
        """The set of branches of the module, which can be modified directly."""
        return self._branches

    @property
    def grouped(self) -> bool:
        """Indicates whether the module is grouped or not."""
        return self._grouped

    @property
    def is_iopin(self) -> bool:
        """Indicates whether the module is an I/O pin."""
        return self._is_iopin

class FlyLine:
    """
    A class to represent a fly line connecting multiple modules in a graphical view. The connections
    are visualized as lines drawn from the centroid of the fly line to the centroids of each connected
    module.
    """
    _view: GraphicalView # The view where the fly line is shown
    _modules: set[Module] # The modules this fly line connects
    _lines: set[QGraphicsLineItem]
    _weight: float # Number of wires
    _visible: bool = True
    _pen: QPen # The pen used to draw the lines (with the corresponding width)

    def __init__(self, view: GraphicalView, modules: set[Module], weight: float, pen_width: float) -> None:
        self._view = view
        self._modules = modules
        self._lines = set[QGraphicsLineItem]()
        self._weight = weight

        self._pen = QPen()
        self._pen.setCosmetic(True)
        self._pen.setWidthF(pen_width)

        for module in self._modules:
            module.add_fly_line(self)

        self.update_fly_line()

    def update_fly_line(self) -> None:
        """Updates the fly lines by redrawing lines between modules or to the centroid."""
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
        line.setPen(self._pen)
        self._lines.add(line)
        self._view.show_item(line)
        if not self._visible:
            line.hide()

    def centroid_modules(self) -> QPointF:
        """Calculates the centroid of the modules this fly line connects."""
        total_area = 0.0
        x, y = 0.0, 0.0

        for mod in self._modules:
            cent = mod.centroid()
            mod_area = mod.area

            x += mod_area * cent.x()
            y += mod_area * cent.y()
            total_area += mod_area

        if total_area != 0:
            return QPointF(x/total_area, y/total_area)
        else:         
            x, y = 0.0, 0.0
            for mod in self._modules:
                cent = mod.centroid()
                x += mod.x()
                y += mod.y()
            num_mod = len(self._modules)
            return QPointF(x/num_mod, y/num_mod)

    def hide(self) -> None:
        """Hides all lines of the fly line."""
        for line in self._lines:
            line.setVisible(False)
        self._visible = False

    def show(self) -> None:
        """Makes all lines of the fly line visible."""
        for line in self._lines:
            line.setVisible(True)
        self._visible = True

    def clear_lines(self) -> None:
        """Removes and clears all the lines from the fly line."""
        for line in self._lines:
            line.scene().removeItem(line)

        self._lines.clear()

    @property
    def modules(self) -> set[Module]:
        "Returns the set of modules the fly line connects."
        return self._modules


MAGNET = 4
class RectObj(QGraphicsRectItem):
    """
    A class that provides a customized rectangle item that you can add to a QGraphicsScene.
    Optionally includes handles for interactive resizing.
    There are special functions for rectangles used as a module trunk and for those 
    representing I/O pins.
    """

    _handles: dict[str,Handle] # key: corner, value: Handle
    _area: float
    _maintain_area: bool = False
    _module: Module | None = None # Module this rect belongs to

    # If it's a trunk:
    _show_trunk_point: bool = False
    _pen_trunk: QPen | None = None # pen used to draw the trunk point
    _text_trunk: QGraphicsTextItem | None = None # module's name

    # If it's a pin:
    _is_iopin: bool = False
    _horizontal: bool | None = None # if it's not a pin it will be None

    def __init__(self, x: float, y: float, w: float, h: float):
        """Creates the rectangle given the center point (x,y) and the dimensions (w,h)."""
        if w == 0:
            w = MIN_PIN_SIZE
        if h == 0:
            h = MIN_PIN_SIZE
        super().__init__(0, 0, w, h)
        self.setPos(x - w/2, y - h/2)
        self._handles = dict[str,Handle]()
        self._area = w * h

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setOpacity(0.4)
        pen = self.pen()
        pen.setCosmetic(True) # Has always the same width
        pen.setWidth(1)
        self.setPen(pen)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Sets the cursor to an open hand when the rectangle is pressed.
        Clears selection of all other items and selects this rectangle exclusively."""
        self.focusItem()
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        for item in self.scene().selectedItems():
            item.setSelected(False)

        self.setSelected(True)

    def itemChange(self, change: QGraphicsRectItem.GraphicsItemChange, value: QPointF):
        """Handles item position changes. When moving, this rectangle aligns itself with its siblings 
        in the module if they are close enough."""
        if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionChange:
            rect = self.rect()
            rect_midp = QPointF(value.x()+ rect.width()/2, value.y() +rect.height()/2)

            if self._module is not None:
                for sibling in self._module.branches|{self._module.trunk} - {self}:
                    sib_midp, sib_rect = sibling.midpoint(), sibling.rect()

                    dist_x = abs(rect_midp.x() - sib_midp.x()) - (rect.width() + sib_rect.width())/2
                    dist_y = abs(rect_midp.y() - sib_midp.y()) - (rect.height() + sib_rect.height())/2

                    if abs(dist_x) < MAGNET and dist_y <= 0:
                        if sib_midp.x() > rect_midp.x(): # sibling at the right
                            rect_midp.setX(sib_midp.x() - (rect.width()+sib_rect.width())/2)
                        else:
                            rect_midp.setX(sib_midp.x() + (rect.width()+sib_rect.width())/2)
                    elif abs(dist_y) < MAGNET and dist_x <= 0:
                        if sib_midp.y() > rect_midp.y(): # sibling is below
                            rect_midp.setY(sib_midp.y() - (rect.height()+sib_rect.height())/2)
                        else:
                            rect_midp.setY(sib_midp.y() + (rect.height()+sib_rect.height())/2)

            self._update_text_position()
            return self.midpoint_to_topleft(rect_midp)
        if change != QGraphicsRectItem.GraphicsItemChange.ItemChildRemovedChange:
            self._update_text_position()
            
        return super().itemChange(change, value)
    
    def copy(self) -> 'RectObj':
        """Returns a copy of the rectangle."""
        center = self.midpoint()
        rect = RectObj(center.x(), center.y(), self.rect().width(), self.rect().height())
        rect.setBrush(self.brush())
        return rect
    

    def midpoint(self) -> QPointF:
        """Returns de center point of the rectangle (in scene coordinates)."""
        point, rect = self.scenePos(), self.rect()

        x = point.x() + rect.width()/2
        y = point.y() + rect.height()/2
        return QPointF(x,y)
    
    def midpoint_to_topleft(self, mid: QPointF) -> QPointF:
        """
        Given the center point of the rectangle, returns its top-left corner in scene
        coordinates.
        
        Args:
            mid: Center point of the rectangle.

        Returns:
            QPointF: The top-left corner if mid is the center (in scene coordinates).
        """
        rect = self.rect()
        x = mid.x() - rect.width()/2
        y = mid.y() - rect.height()/2
        return QPointF(x,y)
    

    def create_handles(self) -> None:
        """Creates Handles for the rectangle and positions them acordingly."""
        if self._is_iopin:
            corners = ["left", "right"] if self._horizontal else ["top", "bottom"]
            is_iopin = True
        else:
            corners = ["top-left", "top-right", "bottom-right", "bottom-left"]
            is_iopin = False

        for corner in corners:
            handle = Handle(corner, self, is_iopin=is_iopin)
            self._handles[corner] = handle
        self.update_handles_position()

    def update_handles_position(self) -> None:
        """Positions the handles at the corners of the rectangle."""
        rect = self.rect()
        
        if self._is_iopin:
            if self._horizontal:
                self._handles["left"].setPos(rect.topLeft())
                self._handles["right"].setPos(rect.bottomRight())
            else:
                self._handles["top"].setPos(rect.topLeft())
                self._handles["bottom"].setPos(rect.bottomRight())
        else:
            self._handles["top-left"].setPos(rect.topLeft())
            self._handles["bottom-right"].setPos(rect.bottomRight())
            self._handles["top-right"].setPos(rect.topRight())
            self._handles["bottom-left"].setPos(rect.bottomLeft())

    def resize_from_handle(self, corner: str, scene_pos: QPointF) -> None:
        """
        Resizes the rectangle based on the new position of the specified corner's handle.
        
        Args:
            corner: Which corner is being dragged ("top-left", "top-right", "bottom-right", "bottom-left",
            "top", "bottom", "left", "right").
            scene_pos: The new mouse position in scene coordinates.
        """
        local_pos = self.mapFromScene(scene_pos)
        new_rect = self.rect()

        if self._is_iopin:
            if corner == "top":
                new_rect.setTop(local_pos.y())
            elif corner == "bottom":
                new_rect.setBottom(local_pos.y())
            elif corner == "left":
                new_rect.setLeft(local_pos.x())
            elif corner == "right":
                new_rect.setRight(local_pos.x())
        else:
            if corner == "top-left":
                new_rect.setTopLeft(local_pos)
            elif corner == "top-right":
                new_rect.setTopRight(local_pos)
            elif corner == "bottom-right":
                new_rect.setBottomRight(local_pos)
            elif corner == "bottom-left":
                new_rect.setBottomLeft(local_pos)

        # Prevent creating a rectangle that is too small
        w, h = new_rect.width(), new_rect.height()
        if self._is_iopin and ( w < MIN_PIN_SIZE or h < MIN_PIN_SIZE):
            return
        if not self._is_iopin and (w < 1 or h < 1):
            return

        self.prepareGeometryChange()
        self.setRect(new_rect)
        self.update_handles_position()
        self._update_text_position()

    def resize_from_handle_maintain_area(self, corner: str, scene_pos: QPointF) -> None:
        """
        Resizes the rectangle based on the new position of the specified corner's handle
        while keeping the same area.

        Args:
            corner: Which corner is being dragged ("top-left", "top-right", "bottom-right", "bottom-left").
            scene_pos: The new mouse position in scene coordinates.
        """
        local_pos = self.mapFromScene(scene_pos)
        new_rect = self.rect()
        area = self.area

        if corner == "top-left":
            last_pos = new_rect.topLeft()
            dx, dy = abs(last_pos.x() - local_pos.x()), abs(last_pos.y() - local_pos.y())
            if dx > dy: # Horizontal change dominates
                new_rect.setLeft(local_pos.x())
                new_rect.setBottom(new_rect.top() + area/new_rect.width())
            else: # Vertical change dominates
                new_rect.setTop(local_pos.y())
                new_rect.setRight(new_rect.left() + area/new_rect.height())

        elif corner == "top-right":
            last_pos = new_rect.topRight()
            dx, dy = abs(last_pos.x() - local_pos.x()), abs(last_pos.y() - local_pos.y())
            if dx > dy:
                new_rect.setRight(local_pos.x())
                new_rect.setBottom(new_rect.top() + area/new_rect.width())
            else:
                new_rect.setTop(local_pos.y())
                new_rect.setLeft(new_rect.right() - area/new_rect.height())

        elif corner == "bottom-right":
            last_pos = new_rect.bottomRight()
            dx, dy = abs(last_pos.x() - local_pos.x()), abs(last_pos.y() - local_pos.y())
            if dx > dy:
                new_rect.setRight(local_pos.x())
                new_rect.setTop(new_rect.bottom() - area/new_rect.width())
            else:
                new_rect.setBottom(local_pos.y())
                new_rect.setLeft(new_rect.right() - area/new_rect.height())

        elif corner == "bottom-left":
            last_pos = new_rect.bottomLeft()
            dx, dy = abs(last_pos.x() - local_pos.x()), abs(last_pos.y() - local_pos.y())
            if dx > dy:
                new_rect.setLeft(local_pos.x())
                new_rect.setTop(new_rect.bottom() - area/new_rect.width())
            else:
                new_rect.setBottom(local_pos.y())
                new_rect.setRight(new_rect.left() + area/new_rect.height())

        # Prevent creating a rectangle that is too small
        if new_rect.width() < 5 or new_rect.height() < 5:
            return

        self.prepareGeometryChange()
        self.setRect(new_rect)
        self.update_handles_position()
        self._update_text_position()

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


    def set_orientation(self) -> None:
        """This function is for I/O Pins only. Determines if the pin is oriented horizontally or not."""
        rect = self.rect()
        w, h = rect.width(), rect.height()

        if w != h:
            if w > h:
                self._horizontal = True
            else:
                self._horizontal = False
        else:
            center = self.midpoint()
            if center.x() == 0: # if it's placed in a corner it will be horizontal by default
                self._horizontal = True
            else:
                self._horizontal = False
        
        return None
    
    def change_orientation(self):
        """For I/O pins. Rotates the rectangle 90º, swaps the handles and modifies the 
        horizontal attribute."""
        assert self.is_iopin
        rect = self.rect()
        w, h = rect.width(), rect.height()

        rect.setWidth(h)
        rect.setHeight(w)
        self.setRect(rect)

        if self._handles:
            if self._horizontal: # Change horizontal -> vertical
                self._handles["top"] = self._handles.pop("left")
                self._handles["top"].change_corner("top")
                
                self._handles["bottom"] = self._handles.pop("right")
                self._handles["bottom"].change_corner("bottom")
            else: # Change vertical -> horizontal
                self._handles["left"] = self._handles.pop("top")
                self._handles["left"].change_corner("left")
                self._handles["right"] = self._handles.pop("bottom")
                self._handles["right"].change_corner("right")

        self._horizontal = not self._horizontal
        self.update_handles_position()

    def set_as_trunk(self) -> None:
        """Marks the rectangle as a trunk, creates a pen for drawing its center point,
        and creates the text item for displaying the module's name."""
        if not self._is_iopin:
            self._show_trunk_point = True
            color = self.brush().color()
            complementary_color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())

            self._pen_trunk = QPen(complementary_color)
            self._pen_trunk.setWidth(12)
            self._pen_trunk.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            self._pen_trunk.setCosmetic(True)
        
        assert self._module is not None
        self._text_trunk = QGraphicsTextItem(self._module.name, self)
        font = QFont("Helvetica", 10, QFont.Weight.Bold)
        self._text_trunk.setFont(font)
        self._text_trunk.setDefaultTextColor(Qt.GlobalColor.black)
        self._text_trunk.setFlag(QGraphicsTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self._text_trunk.setPos(self.rect().center())
        self._text_trunk.adjustSize()

    def _update_text_position(self) -> None:
        """If the item has text (it's a trunk), centers the text in the middle of the rectangle.
        For I/O pins adjusts its rotation if it's placed at the scene borders."""
        if self._text_trunk:
            center = self.rect().center()
            self._text_trunk.setPos(center)

            if self._is_iopin:
                pos = self.pos()
                scene = self.scene()
                if abs(pos.x()) < 2:
                    self._text_trunk.setRotation(0)
                    self._text_trunk.setX(center.x()-12)
                elif abs(pos.y()) < 2:
                    self._text_trunk.setRotation(300)
                    self._text_trunk.setY(center.y()-2)
                elif scene and abs(pos.y() - scene.sceneRect().bottom()) < 2:
                    self._text_trunk.setRotation(45)
                elif scene and abs(pos.x() - scene.sceneRect().right()) < 2:
                    self._text_trunk.setRotation(0)

    def show_text(self, show: bool) -> None:
        """Shows or hides the trunk's text depending on the value of 'show'."""
        if self._text_trunk:
            self._text_trunk.setVisible(show)
    
    def show_trunk_point(self, show: bool) -> None:
        """Shows or hides the trunk's point depending on the value of 'show'."""
        if show != self._show_trunk_point:
            self._show_trunk_point = show
            self.update()

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, /, widget: QWidget | None = ...) -> None: # type: ignore
        """Draws the rectangle. If it's a trunk and '_show_trunk_point' is enabled, draws a small center 
        point with the complementary color."""
        super().paint(painter, option, widget) # Draw the rectangle as always

        if self._show_trunk_point and self._pen_trunk: #pins don't show this point
            painter.setPen(self._pen_trunk)
            painter.setOpacity(0.8)
            center = self.rect().center()
            painter.drawPoint(center)

    def set_module(self, module: Module):
        """Sets the module this rectangle belongs to."""
        self._module = module

    @property
    def area(self) -> float:
        """Returns the area of the rectangle."""
        return self._area
    
    @property
    def maintain_area(self) -> bool:
        """Returns whether the rectangle's area should be preserved when resizing."""
        return self._maintain_area

    @maintain_area.setter
    def maintain_area(self, maintain: bool) -> None:
        """Set whether the area should be preserved when resizing."""
        self._maintain_area = maintain

    @property
    def is_iopin(self) -> bool:
        """Returns whether the rectangle is part of an I/O pin."""
        return self._is_iopin
    
    @is_iopin.setter
    def is_iopin(self, is_iopin: bool) -> None:
        """Set whether the rectangle is part of an I/O pin."""
        self._is_iopin = is_iopin

    @property
    def is_horizontal(self) -> bool:
        """Returns whether the rectangle is horizontal or not. This property is only for soft/hard pins."""
        assert self._horizontal is not None
        return self._horizontal

HANDLE_SIZE = 16
HANDLE_IO_PIN = 10
class Handle(QGraphicsRectItem):
    """
    A class to create a handle used to resize a parent RectObj.
    This handle appears as a semi-transparent square positioned at the specified
    corner of the parent item.
    """  
    _corner: str

    def __init__(self, corner: str, parent: RectObj, is_iopin: bool = False):   
        """Initializes a new Handle instance for resizing the given parent RectObj."""
        if corner in "top-left":
            if is_iopin:
                x, y =  -HANDLE_IO_PIN/2, -HANDLE_IO_PIN/2
            else:
                x, y = 0, 0
        elif corner == "top-right":
            x, y = -HANDLE_SIZE, 0
        elif corner in "bottom-right":
            if is_iopin:
                x, y = -HANDLE_IO_PIN/2, -HANDLE_IO_PIN/2
            else:
                x, y = -HANDLE_SIZE, -HANDLE_SIZE
        elif corner == "bottom-left":
            x, y = 0, -HANDLE_SIZE
        else:
            assert False, "Incorrect corner"

        if is_iopin:
            super().__init__(x, y, HANDLE_IO_PIN, HANDLE_IO_PIN)
        else:
            super().__init__(x, y, HANDLE_SIZE, HANDLE_SIZE)

        self._corner = corner
        self.setParentItem(parent)
        self.setAcceptHoverEvents(True)

        self.setBrush(Qt.GlobalColor.transparent)
        self.setOpacity(0.1)
        self.setFlags(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable 
                      | QGraphicsRectItem.GraphicsItemFlag.ItemSendsScenePositionChanges 
                      | QGraphicsRectItem.GraphicsItemFlag.ItemIgnoresTransformations)

    def itemChange(self, change: QGraphicsRectItem.GraphicsItemChange, value: QPointF):
        """Handles item position changes and resizes the parent RectObj accordingly."""
        if change == QGraphicsRectItem.GraphicsItemChange.ItemScenePositionHasChanged:
            rect = self.parentItem()
            assert isinstance(rect, RectObj)

            if rect.maintain_area:
                rect.resize_from_handle_maintain_area(self._corner, value)
            else:
                rect.resize_from_handle(self._corner, value)
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
        """Changes the cursor to a resize icon depending on the handle's corner."""
        if self._corner in ("top-left", "bottom-right"):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif self._corner in ("top", "bottom"):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif self._corner in ("left", "right"):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else: # "top.right", "bottom-lelft"
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent):
        """Restores the default cursor when the mouse leaves the handle area."""
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def change_corner(self, corner: str) -> None:
        """Changes the corner attribute to the given corner (or side)."""
        self._corner = corner

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Clears selection of all other items and selects this handle exclusively."""
        self.focusItem()
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        for item in self.scene().selectedItems():
            item.setSelected(False)

        self.setSelected(True)
        return super().mousePressEvent(event)