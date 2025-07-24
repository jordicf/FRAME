
def midpoint_to_topleft(x: float, y:float, w: float, h:float) -> tuple[float,float]:

    return (x - w/2, y - h/2)

def topleft_to_midpoint(x: float, y:float, w: float, h:float) -> tuple[float,float]:

    return (x + w/2, y + h/2)
