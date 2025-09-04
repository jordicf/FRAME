# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).
import os
import ctypes
from typing import Type


class BOX(ctypes.Structure):
    _fields_ = ("x1", ctypes.c_double), \
               ("y1", ctypes.c_double), \
               ("x2", ctypes.c_double), \
               ("y2", ctypes.c_double), \
               ("p", ctypes.c_double)


class GreedyManager:

    def __init__(self):
        # path = os.path.abspath(os.path.join(os.path.dirname(__file__), "rect_greedy.pyd"))
        path = "C:/Users/Lenovo/Documents/GitHub/FRAME/tools/rect/rect_greedy.pyd"
        self.mylib = ctypes.CDLL(path)
        self.mylib.find_best_box.restype = BOX
        self.mylib.find_best_box.argtypes = [ctypes.POINTER(BOX), ctypes.c_double, ctypes.c_double, ctypes.c_long, ctypes.c_double]

    def find_best_box(self, ww: float, hh: float, nb: int, pr: float,
                      inboxes: list[tuple[float, float, float, float, float]]) -> \
            tuple[float, float, float, float, float]:
        assert(nb == len(inboxes))

        # First, we cast the inputs into c types
        boxarr: Type[ctypes.Array[BOX]] = BOX * nb
        inboxrecast = ctypes.cast(boxarr(*(map(lambda t: BOX(t[0], t[1], t[2], t[3], t[4]), inboxes))), ctypes.POINTER(BOX))

        w = ctypes.c_double(ww)
        h = ctypes.c_double(hh)
        n = ctypes.c_long(nb)
        p = ctypes.c_double(pr)

        # Then, we call the solver
        box: BOX = self.mylib.find_best_box(inboxrecast, w, h, n, p)

        # Finally, we return the result
        return box.x1, box.y1, box.x2, box.y2, box.p
