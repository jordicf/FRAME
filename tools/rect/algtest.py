problemInput: list[tuple[float, float, float, float]] = [
    (0, 0, 2, 2),
    (2, 0, 4, 2),
    (0, 2, 2, 4),
    (2, 2, 4, 4),
    (4, 0, 8, 4),
    (0, 4, 2, 6),
    (2, 4, 3, 5),
    (3, 4, 4, 5),
    (2, 5, 3, 6),
    (3, 5, 4, 6),
    (4, 4, 6, 6),
    (6, 4, 8, 6),
    (0, 6, 2, 8),
    (2, 6, 3, 7),
    (3, 6, 4, 7),
    (2, 7, 3, 8),
    (3, 7, 4, 8),
    (4, 6, 6, 8),
    (6, 6, 8, 8)
]

max_coord: float = -100000
for b in problemInput:
    (xx1, yy1, xx2, yy2) = b
    if xx2 > max_coord:
        max_coord = xx2
    if yy2 > max_coord:
        max_coord = yy2

rep_count: int = 0


def report(bl: tuple[float, float, float, float]) -> None:
    global rep_count
    print(rep_count, ":", bl)
    rep_count += 1


block = tuple[float, float, float, float]
hmap: dict[tuple[float, float, float], block] = {}
vmap: dict[tuple[float, float, float], block] = {}


def join(b1: block, b2: block) -> block:
    # Returns the bounding box of the two blocks
    (b1x1, b1y1, b1x2, b1y2) = b1
    (b2x1, b2y1, b2x2, b2y2) = b2
    return min(b1x1, b2x1), min(b1y1, b2y1), max(b1x2, b2x2), max(b1y2, b2y2)


def insert_block(b1: block, horizontal: bool = True) -> None:
    (x1, y1, x2, y2) = b1
    if (x2, y1, y2) not in hmap or x1 > hmap[x2, y1, y2][0]:
        hmap[x2, y1, y2] = b1
    if (y2, x1, x2) not in vmap or y1 > vmap[y2, x1, x2][1]:
        vmap[y2, x1, x2] = b1

    report(b1)

    wall: bool = False
    if horizontal:
        if (x1, y1, y2) in hmap:
            insert_block(join(b1, hmap[x1, y1, y2]), True)
        else:
            wall = True
    if (y1, x1, x2) in vmap:
        insert_block(join(b1, vmap[y1, x1, x2]), wall)


problemInput.sort(key=lambda x: x[3] * max_coord + x[2])
for b in problemInput:
    insert_block(b)
