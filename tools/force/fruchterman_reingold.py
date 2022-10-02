# (c) Marçal Comajoan Cara 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from itertools import combinations

from frame.die.die import Die
from frame.geometry.geometry import Point
from tools.draw.draw import get_floorplan_plot


def fruchterman_reingold_layout(die: Die, verbose: bool = False, visualize: str | None = None,
                                max_iter: int = 100) -> Die:
    assert die.netlist is not None, "No netlist associated to the die"

    vis_imgs = []

    mod2idx = {mod: idx for idx, mod in enumerate(die.netlist.modules)}

    t = max(die.width, die.height) * 0.1
    dt = t / (max_iter + 1)

    k = (die.width * die.height / die.netlist.num_modules)**(1 / 2)

    def f_att(x, w):
        return w * x**2 / k

    def f_rep(x, w):
        return w * k**2 / max(x, 0.01)

    def die_repelling(p: Point, w: float) -> Point:
        repelling = Point(0, 0)
        if p.x < -die.width / 2 + die.width / 10:
            repelling += Point(1, 0) * f_rep(p.x + die.width / 2, w)
        if p.x > die.width / 2 - die.width / 10:
            repelling += Point(-1, 0) * f_rep(die.width / 2 - p.x, w)
        if p.y < -die.height / 2 + die.height / 10:
            repelling += Point(0, 1) * f_rep(p.y + die.height / 2, w)
        if p.y > die.height / 2 - die.height / 10:
            repelling += Point(0, -1) * f_rep(die.height / 2 -  p.y, w)
        return repelling

    pos: list[Point] = [module.center - Point(die.width, die.height) / 2 if module.center is not None else Point()
                        for module in die.netlist.modules]  # The die is recentered to the origin
    disp = [Point()] * die.netlist.num_modules
    for i in range(max_iter):
        for v in range(die.netlist.num_modules):
            disp[v] = Point()
            for u in range(die.netlist.num_modules):
                if u != v:
                    diff = pos[v] - pos[u]
                    diff_norm = max(diff.norm(), 0.01)
                    disp[v] += diff / diff_norm * f_rep(diff_norm, die.netlist.modules[v].area()) \
                        + die_repelling(pos[v], die.netlist.modules[v].area())

        for hyperedge in die.netlist.edges:
            for v_mod, u_mod in combinations(hyperedge.modules, 2):
                v, u = mod2idx[v_mod], mod2idx[u_mod]
                diff = pos[v] - pos[u]
                diff_norm = max(diff.norm(), 0.01)
                disp[v] -= diff / diff_norm * f_att(diff_norm, hyperedge.weight)
                disp[u] += diff / diff_norm * f_att(diff_norm, hyperedge.weight)

        for v in range(die.netlist.num_modules):
            if not die.netlist.modules[v].is_fixed:
                disp_norm = max(disp[v].norm(), 0.01)
                pos[v] += disp[v] / disp_norm * min(disp_norm, t)
                pos[v].x = min(die.width / 2, max(-die.width / 2, pos[v].x))
                pos[v].y = min(die.height / 2, max(-die.height / 2, pos[v].y))

        t -= dt

        if verbose:
            print(i, end=" ", flush=True)

        if visualize is not None:
            for v, module in enumerate(die.netlist.modules):
                module.center = pos[v] + Point(die.width, die.height) / 2
            vis_imgs.append(get_floorplan_plot(die.netlist, die.bounding_box.shape))

    if verbose:
        print("\nAlgorithm completed.")

    if visualize is None:
        for v, module in enumerate(die.netlist.modules):
            module.center = pos[v] + Point(die.width, die.height) / 2
    else:
        vis_imgs[0].save(f"{visualize}.gif", save_all=True, append_images=vis_imgs[1:], duration=100)

    return die
