# (c) Ylham Imam, 2025 — CasADi port
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from __future__ import annotations

import sys
import math
import argparse
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import casadi as ca

from frame.die.die import Die
from frame.netlist.netlist import Netlist
from frame.geometry.geometry import Rectangle

# Import matplotlib for visualization (optional)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# -------------------------
# Types
# -------------------------
BoxType = tuple[float, float, float, float]
InputModule = tuple[BoxType, list[BoxType], list[BoxType], list[BoxType], list[BoxType]]
OptionalList = dict[int, float]
OptionalMatrix = dict[int, OptionalList]
HyperEdge = tuple[float, list[int]]
HyperGraph = list[tuple[float, list[int], list[str]]]


def parse_options(prog: Optional[str] = None, args: Optional[list[str]] = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="A tool for module legalization (CasADi)",
        usage="%(prog)s [options]",
    )
    parser.add_argument("netlist", type=str, help="Input netlist (.yaml)")
    parser.add_argument("die", type=str, help="Input die (.yaml)")
    parser.add_argument("--max_ratio", type=float, default=3.0, help="Max aspect ratio")
    parser.add_argument("--num_iter", type=int, default=15, help="Number of iterations")
    parser.add_argument("--radius", type=float, default=1.0, help="No-overlap distance radius multiplier")
    parser.add_argument("--wl_mult", type=float, default=1.0, help="HPWL weight multiplier")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--t0", type=float, default=0.3, help="Initial annealing temperature (unused)")
    parser.add_argument("--dt", type=float, default=0.9, help="Temperature factor (unused)")
    parser.add_argument("--dcost", type=float, default=1e-5, help="Delta cost (unused)")
    parser.add_argument("--outfile", type=str, default=None, help="Output YAML")
    parser.add_argument("--palette_seed", type=int, default=None)
    parser.add_argument("--tau_initial", type=float, default=None, help="Initial tau for soft constraints")
    parser.add_argument("--tau_decay", type=float, default=0.3, help="Tau decay factor per step")
    parser.add_argument("--otol_initial", type=float, default=1e-1)
    parser.add_argument("--otol_final", type=float, default=1e-4)
    parser.add_argument("--rtol_initial", type=float, default=1e-1)
    parser.add_argument("--rtol_final", type=float, default=1e-4)
    parser.add_argument("--tol_decay", type=float, default=0.5)
    parser.add_argument("--plot", action="store_true", help="Enable visualization of each iteration")
    parser.add_argument("--plot_dir", type=str, default="plots", help="Directory for saving plots (default: plots)")
    return vars(parser.parse_args(args))


def netlist_to_utils(netlist: Netlist):
    ml: list[InputModule] = []
    al: list[float] = []
    xl: OptionalMatrix = {}
    yl: OptionalMatrix = {}
    wl: OptionalMatrix = {}
    hl: OptionalMatrix = {}
    mod_map: dict[str, int] = {}
    og_names: list[str] = []
    terminal_map: dict[str, tuple[float, float]] = {}

    # Terminals (io_pin): treat coordinates as constants
    # Following legalizer.py: terminals are is_iopin, not just is_fixed
    for module in netlist.modules:
        if module.is_iopin:
            if hasattr(module, "center") and module.center:
                if hasattr(module.center, "x") and hasattr(module.center, "y"):
                    terminal_map[module.name] = (float(module.center.x), float(module.center.y))
                else:
                    terminal_map[module.name] = (float(module.center[0]), float(module.center[1]))
            else:
                rect = module.rectangles[0]
                terminal_map[module.name] = (float(rect.center.x), float(rect.center.y))
            continue

        # Normal modules (including fixed and hard)
        mod_map[module.name] = len(ml)
        og_names.append(module.name)
        b: InputModule = ((0, 0, 0, 0), [], [], [], [])
        trunk_defined = False
        for rect in module.rectangles:
            r = (rect.center.x, rect.center.y, rect.shape.w, rect.shape.h)
            if rect.location == Rectangle.StropLocation.TRUNK:
                b = (r, b[1], b[2], b[3], b[4])
                trunk_defined = True
            elif rect.location == Rectangle.StropLocation.NORTH:
                b[1].append(r)
            elif rect.location == Rectangle.StropLocation.SOUTH:
                b[2].append(r)
            elif rect.location == Rectangle.StropLocation.EAST:
                b[3].append(r)
            elif rect.location == Rectangle.StropLocation.WEST:
                b[4].append(r)
            elif not trunk_defined:
                b = (r, b[1], b[2], b[3], b[4])
                trunk_defined = True
            else:
                b[1].append(r)
        # Following legalizer.py lines 1582-1602
        if module.is_hard:
            xl[len(ml)] = {}
            yl[len(ml)] = {}
            wl[len(ml)] = {}
            hl[len(ml)] = {}
        if module.is_fixed:
            # For fixed modules: store position for trunk
            if len(ml) not in xl:
                xl[len(ml)] = {}
                yl[len(ml)] = {}
            xl[len(ml)][0] = b[0][0]
            yl[len(ml)][0] = b[0][1]
        if module.is_hard:
            wl[len(ml)][0] = b[0][2]
            hl[len(ml)][0] = b[0][3]
            i = 1
            for q in range(1, 5):
                bq = b[q]
                if isinstance(bq, list):
                    for x, y, w, h in bq:
                        xl[len(ml)][i] = x
                        yl[len(ml)][i] = y
                        wl[len(ml)][i] = w
                        hl[len(ml)][i] = h
                        i += 1
        ml.append(b)
        al.append(module.area())

    hyper: HyperGraph = []
    for edge in netlist.edges:
        modules = []
        terminals = []
        weight = edge.weight
        for e_mod in edge.modules:
            if e_mod.name in mod_map:
                modules.append(mod_map[e_mod.name])
            elif e_mod.name in terminal_map:
                terminals.append(e_mod.name)
        if modules:
            hyper.append((weight, modules, terminals))

    return ml, al, xl, yl, wl, hl, hyper, og_names, terminal_map


def compute_options(options):
    die = Die(options["die"])
    die_width: float = die.width
    die_height: float = die.height
    netlist = Netlist(options["netlist"])
    ml, al, xl, yl, wl, hl, hyper, og_names, terminal_map = netlist_to_utils(netlist)
    return (
        ml,
        al,
        xl,
        yl,
        wl,
        hl,
        die_width,
        die_height,
        hyper,
        options["max_ratio"],
        og_names,
        terminal_map,
    )


@dataclass
class ModuleVars:
    """
    Stores variables for all rectangles in a module (trunk + branches)
    Similar to ModelModule in legalizer.py
    """
    x: list[ca.MX]  # x-coordinates of all rectangles [trunk, north_branches, south_branches, east_branches, west_branches]
    y: list[ca.MX]  # y-coordinates
    w: list[ca.MX]  # widths
    h: list[ca.MX]  # heights
    N: list[int]    # indices of North branches
    S: list[int]    # indices of South branches
    E: list[int]    # indices of East branches
    W: list[int]    # indices of West branches
    c: int          # total number of rectangles
    area_expr: ca.MX  # sum of all rectangle areas
    x_sum: ca.MX      # for center of mass calculation
    y_sum: ca.MX      # for center of mass calculation


class CasadiLegalizer:
    def __init__(
        self,
        ml: list[InputModule],
        al: list[float],
        xl: OptionalMatrix,
        yl: OptionalMatrix,
        wl: OptionalMatrix,
        hl: OptionalMatrix,
        die_width: float,
        die_height: float,
        hyper: HyperGraph,
        max_ratio: float,
        og_names: list[str],
        wl_mult: float,
        tau_initial: Optional[float],
        tau_decay: float,
        otol_initial: float,
        otol_final: float,
        rtol_initial: float,
        rtol_final: float,
        tol_decay: float,
        terminal_map: dict[str, tuple[float, float]],
    ) -> None:
        self.ml = ml
        self.al = al
        self.xl = xl
        self.yl = yl
        self.wl = wl
        self.hl = hl
        self.dw = die_width
        self.dh = die_height
        self.hyper = hyper
        self.max_ratio = max_ratio
        self.og_names = og_names
        self.wl_mult = wl_mult
        self.tau_initial = tau_initial if tau_initial is not None else sum(al) / 1.0
        self.tau_decay = tau_decay
        self.otol_initial = otol_initial
        self.otol_final = otol_final
        self.rtol_initial = rtol_initial
        self.rtol_final = rtol_final
        self.tol_decay = tol_decay
        self.terminal_map = terminal_map  # terminals as constants

        # Store initial data for rebuilding problem in each iteration
        self._initial_ml = ml
        self._initial_al = al
        self._initial_xl = xl
        self._initial_yl = yl
        self._initial_wl = wl
        self._initial_hl = hl
        self._initial_og_names = og_names

        # Define symbolic variables once at initialization
        self.vars: list[ca.MX] = []
        self.lbx: list[float] = []
        self.ubx: list[float] = []
        self.x0: list[float] = []  # will be updated as warm-start
        self.modules: list[ModuleVars] = []

        # Initialize module variables and bounds for ALL rectangles
        # Following legalizer.py architecture: each module has lists of variables for all rectangles
        for idx, trunk_data in enumerate(self.ml):
            (x0, y0, w0, h0), Nb, Sb, Eb, Wb = trunk_data
            
            # Trunk (rectangle 0)
            trunk_x, trunk_y, trunk_w, trunk_h = self._define_rect_vars((x0, y0, w0, h0))
            x_list = [trunk_x]
            y_list = [trunk_y]
            w_list = [trunk_w]
            h_list = [trunk_h]
            
            N_indices = []
            S_indices = []
            E_indices = []
            W_indices = []
            
            rect_count = 1  # Start at 1 (trunk is 0)
            
            # Add North branches
            for (bx, by, bw, bh) in Nb:
                bx_var, by_var, bw_var, bh_var = self._define_rect_vars((bx, by, bw, bh))
                x_list.append(bx_var)
                y_list.append(by_var)
                w_list.append(bw_var)
                h_list.append(bh_var)
                N_indices.append(rect_count)
                rect_count += 1
            
            # Add South branches
            for (bx, by, bw, bh) in Sb:
                bx_var, by_var, bw_var, bh_var = self._define_rect_vars((bx, by, bw, bh))
                x_list.append(bx_var)
                y_list.append(by_var)
                w_list.append(bw_var)
                h_list.append(bh_var)
                S_indices.append(rect_count)
                rect_count += 1
            
            # Add East branches
            for (bx, by, bw, bh) in Eb:
                bx_var, by_var, bw_var, bh_var = self._define_rect_vars((bx, by, bw, bh))
                x_list.append(bx_var)
                y_list.append(by_var)
                w_list.append(bw_var)
                h_list.append(bh_var)
                E_indices.append(rect_count)
                rect_count += 1
            
            # Add West branches
            for (bx, by, bw, bh) in Wb:
                bx_var, by_var, bw_var, bh_var = self._define_rect_vars((bx, by, bw, bh))
                x_list.append(bx_var)
                y_list.append(by_var)
                w_list.append(bw_var)
                h_list.append(bh_var)
                W_indices.append(rect_count)
                rect_count += 1
            
            # Calculate total area and center of mass
            area_expr = ca.MX(0)
            x_sum = ca.MX(0)
            y_sum = ca.MX(0)
            for i in range(rect_count):
                rect_area = w_list[i] * h_list[i]
                area_expr = area_expr + rect_area
                x_sum = x_sum + x_list[i] * rect_area
                y_sum = y_sum + y_list[i] * rect_area
            
            self.modules.append(ModuleVars(
                x=x_list, y=y_list, w=w_list, h=h_list,
                N=N_indices, S=S_indices, E=E_indices, W=W_indices,
                c=rect_count,
                area_expr=area_expr,
                x_sum=x_sum,
                y_sum=y_sum
            ))

        self.x_sym = ca.vertcat(*self.vars)

        # Define parameters for dynamic updates
        self.tau_param = ca.MX.sym("tau_param", 1) # scalar tau
        # Mask for active inter-module overlap constraints
        num_inter_module_pairs = len(self.modules) * (len(self.modules) - 1) // 2
        self.active_mask_param = ca.MX.sym("active_mask_param", num_inter_module_pairs) # binary mask
        self.step_cap_param = ca.MX.sym("step_cap_param", 1) # scalar step cap for trust region
        self.params = ca.vertcat(self.tau_param, self.active_mask_param, self.step_cap_param)

        self.nlp_initialized = False

    def _define_rect_vars(self, rect: BoxType) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX]:
        x = ca.MX.sym("x", 1)
        y = ca.MX.sym("y", 1)
        w = ca.MX.sym("w", 1)
        h = ca.MX.sym("h", 1)
        self.vars.extend([x, y, w, h])
        self.lbx.extend([0.0, 0.0, 0.1, 0.1])
        self.ubx.extend([self.dw, self.dh, self.dw, self.dh])
        self.x0.extend([rect[0], rect[1], max(rect[2], 0.1), max(rect[3], 0.1)])
        return x, y, w, h

    def _rebuild_solver_instance(self,
                                 current_tau_val: float,
                                 active_mask_vals: list[float],
                                 prev_vals: Optional[list[float]],
                                 step_cap_val: Optional[float],
                                 otol: float,
                                 rtol: float,
                                 verbose: bool = False) -> None:
        g: list[ca.MX] = []
        lbg: list[float] = []
        ubg: list[float] = []

        # 1. Bounds constraints (die) - for ALL rectangles
        for m in self.modules:
            for i in range(m.c):
                g.append(m.x[i] - 0.5 * m.w[i]) ; lbg.append(0.0) ; ubg.append(ca.inf)
                g.append(m.y[i] - 0.5 * m.h[i]) ; lbg.append(0.0) ; ubg.append(ca.inf)
                g.append(m.x[i] + 0.5 * m.w[i] - self.dw) ; lbg.append(-ca.inf) ; ubg.append(0.0)
                g.append(m.y[i] + 0.5 * m.h[i] - self.dh) ; lbg.append(-ca.inf) ; ubg.append(0.0)

        # 2. Minimal area requirements (skip if al ~ 0)
        for i, m in enumerate(self.modules):
            if self.al[i] > 1e-9:
                g.append(m.area_expr) ; lbg.append(self.al[i]) ; ubg.append(ca.inf)

        # 3. Aspect ratio constraint via thin(w,h) >= thin(max_ratio,1) - for ALL rectangles
        def thin(w: ca.MX, h: ca.MX) -> ca.MX:
            return (w * h) / (w * w + h * h )
        thin_min = (self.max_ratio * 1.0) / (self.max_ratio * self.max_ratio + 1.0)
        for m in self.modules:
            for i in range(m.c):
                g.append(thin(m.w[i], m.h[i])) ; lbg.append(thin_min) ; ubg.append(ca.inf)
        
        # 3.5. Attachment constraints: branches must attach to trunk
        # Following legalizer.py: add_rect_north/south/east/west
        for m in self.modules:
            trunk_x, trunk_y, trunk_w, trunk_h = m.x[0], m.y[0], m.w[0], m.h[0]
            
            # North branches: yv = trunk_y + 0.5*trunk_h + 0.5*hv
            for rect_idx in m.N:
                bx, by, bw, bh = m.x[rect_idx], m.y[rect_idx], m.w[rect_idx], m.h[rect_idx]
                # Attachment
                g.append(by - (trunk_y + 0.5 * trunk_h + 0.5 * bh))
                lbg.append(0.0); ubg.append(0.0)
                # Border: xv >= trunk_x - 0.5*trunk_w + 0.5*wv
                g.append(bx - (trunk_x - 0.5 * trunk_w + 0.5 * bw))
                lbg.append(0.0); ubg.append(ca.inf)
                # Border: xv <= trunk_x + 0.5*trunk_w - 0.5*wv
                g.append((trunk_x + 0.5 * trunk_w - 0.5 * bw) - bx)
                lbg.append(0.0); ubg.append(ca.inf)
            
            # South branches: yv = trunk_y - 0.5*trunk_h - 0.5*hv
            for rect_idx in m.S:
                bx, by, bw, bh = m.x[rect_idx], m.y[rect_idx], m.w[rect_idx], m.h[rect_idx]
                g.append(by - (trunk_y - 0.5 * trunk_h - 0.5 * bh))
                lbg.append(0.0); ubg.append(0.0)
                g.append(bx - (trunk_x - 0.5 * trunk_w + 0.5 * bw))
                lbg.append(0.0); ubg.append(ca.inf)
                g.append((trunk_x + 0.5 * trunk_w - 0.5 * bw) - bx)
                lbg.append(0.0); ubg.append(ca.inf)
            
            # East branches: xv = trunk_x + 0.5*trunk_w + 0.5*wv
            for rect_idx in m.E:
                bx, by, bw, bh = m.x[rect_idx], m.y[rect_idx], m.w[rect_idx], m.h[rect_idx]
                g.append(bx - (trunk_x + 0.5 * trunk_w + 0.5 * bw))
                lbg.append(0.0); ubg.append(0.0)
                g.append(by - (trunk_y - 0.5 * trunk_h + 0.5 * bh))
                lbg.append(0.0); ubg.append(ca.inf)
                g.append((trunk_y + 0.5 * trunk_h - 0.5 * bh) - by)
                lbg.append(0.0); ubg.append(ca.inf)
            
            # West branches: xv = trunk_x - 0.5*trunk_w - 0.5*wv
            for rect_idx in m.W:
                bx, by, bw, bh = m.x[rect_idx], m.y[rect_idx], m.w[rect_idx], m.h[rect_idx]
                g.append(bx - (trunk_x - 0.5 * trunk_w - 0.5 * bw))
                lbg.append(0.0); ubg.append(0.0)
                g.append(by - (trunk_y - 0.5 * trunk_h + 0.5 * bh))
                lbg.append(0.0); ubg.append(ca.inf)
                g.append((trunk_y + 0.5 * trunk_h - 0.5 * bh) - by)
                lbg.append(0.0); ubg.append(ca.inf)
        
        # 3.6. Intra-module non-overlap: consecutive branches in same direction must not overlap
        for m in self.modules:
            # North branches: sort by x, check consecutive pairs don't overlap
            if len(m.N) > 1:
                for i in range(len(m.N) - 1):
                    idx1, idx2 = m.N[i], m.N[i+1]
                    # x1 + 0.5*w1 <= x2 - 0.5*w2
                    g.append(m.x[idx1] + 0.5 * m.w[idx1] - m.x[idx2] + 0.5 * m.w[idx2])
                    lbg.append(-ca.inf); ubg.append(0.0)
            
            # South branches
            if len(m.S) > 1:
                for i in range(len(m.S) - 1):
                    idx1, idx2 = m.S[i], m.S[i+1]
                    g.append(m.x[idx1] + 0.5 * m.w[idx1] - m.x[idx2] + 0.5 * m.w[idx2])
                    lbg.append(-ca.inf); ubg.append(0.0)
            
            # East branches: sort by y
            if len(m.E) > 1:
                for i in range(len(m.E) - 1):
                    idx1, idx2 = m.E[i], m.E[i+1]
                    g.append(m.y[idx1] + 0.5 * m.h[idx1] - m.y[idx2] + 0.5 * m.h[idx2])
                    lbg.append(-ca.inf); ubg.append(0.0)
            
            # West branches
            if len(m.W) > 1:
                for i in range(len(m.W) - 1):
                    idx1, idx2 = m.W[i], m.W[i+1]
                    g.append(m.y[idx1] + 0.5 * m.h[idx1] - m.y[idx2] + 0.5 * m.h[idx2])
                    lbg.append(-ca.inf); ubg.append(0.0)

        # 4. Inter-module non-overlap: check ALL rectangle pairs between different modules
        # Following legalizer.py: check all combinations of rectangles
        def smax(a: ca.MX, b: ca.MX, tau: ca.MX) -> ca.MX:
            return 0.5 * (a + b + ca.sqrt((a - b) * (a - b) + 4 * tau * tau))

        pair_idx_counter = 0
        inter_module_constraints_added = 0
        inter_module_pairs_skipped = 0
        
        for i in range(len(self.modules)):
            for j in range(i + 1, len(self.modules)):
                if active_mask_vals[pair_idx_counter] > 0.5: # if active (value > 0.5 means 1)
                    mi, mj = self.modules[i], self.modules[j]
                    for rect_i in range(mi.c):
                        for rect_j in range(mj.c):
                            xi, yi, wi, hi = mi.x[rect_i], mi.y[rect_i], mi.w[rect_i], mi.h[rect_i]
                            xj, yj, wj, hj = mj.x[rect_j], mj.y[rect_j], mj.w[rect_j], mj.h[rect_j]
                            t1 = (xi - xj) * (xi - xj) - 0.25 * (wi + wj) * (wi + wj)
                            t2 = (yi - yj) * (yi - yj) - 0.25 * (hi + hj) * (hi + hj)
                            g.append(smax(t1, t2, self.tau_param))
                            lbg.append(0.0) ; ubg.append(ca.inf)
                            inter_module_constraints_added += 1
                else:
                    inter_module_pairs_skipped += 1
                pair_idx_counter += 1
        
        if verbose:
            print(f"[Section 4] Inter-module non-overlap constraints:")
            print(f"  - Total module pairs: {pair_idx_counter}")
            print(f"  - Active module pairs: {pair_idx_counter - inter_module_pairs_skipped}")
            print(f"  - Skipped module pairs: {inter_module_pairs_skipped}")
            print(f"  - Total rectangle-pair constraints added: {inter_module_constraints_added}")

        # 5. Trust-Region constraints: |x - x_prev| <= step_cap
        if prev_vals is not None and step_cap_val is not None and step_cap_val > 0:
            for k, v_sym in enumerate(self.vars):
                pv = float(prev_vals[k]) # pv comes from last_x, which is an x_sym solution
                # v_sym - (pv + step_cap_val) <= 0
                g.append(v_sym - (pv + self.step_cap_param))
                lbg.append(-ca.inf) ; ubg.append(0.0)
                # (pv - step_cap_val) - v_sym <= 0  <=> v_sym >= pv - step_cap_val
                g.append((pv - self.step_cap_param) - v_sym)
                lbg.append(-ca.inf) ; ubg.append(0.0)

        # 0. Fixed module constraints
        # For fixed modules, add equality constraints for position
        for i, m in enumerate(self.modules):
            # Check if this module is fixed
            has_fixed_position = (i in self.xl and 0 in self.xl[i] and
                                   i in self.yl and 0 in self.yl[i])
            
            if has_fixed_position:
                # Fix trunk position
                g.append(m.x[0] - self.xl[i][0])
                lbg.append(0.0); ubg.append(0.0)
                g.append(m.y[0] - self.yl[i][0])
                lbg.append(0.0); ubg.append(0.0)
                
                if verbose:
                    module_name = self.og_names[i] if i < len(self.og_names) else f"Module_{i}"
                    print(f"[Section 0] Fixed module {module_name} at ({self.xl[i][0]:.2f}, {self.yl[i][0]:.2f})")
        
        # Objective: LSE(HPWL) with terminals as constants
        def stable_logsumexp(vals: list[ca.MX], a: float) -> ca.MX:
            ax = [a * v for v in vals]
            m = ax[0]
            for v in ax[1:]:
                m = ca.fmax(m, v)
            s = 0
            for v in ax:
                s = s + ca.exp(v - m)
            return m + ca.log(s)

        alpha = 1.0
        obj_terms: list[ca.MX] = []

        def module_center(m: ModuleVars) -> tuple[ca.MX, ca.MX]:
            # Center of mass: weighted average of all rectangles
            area_safe = m.area_expr + 1e-10
            cx = m.x_sum / area_safe
            cy = m.y_sum / area_safe
            return cx, cy

        for weight, vertices, terminals in self.hyper:
            pts_x: list[ca.MX] = []
            pts_y: list[ca.MX] = []
            for i in vertices:
                cx, cy = module_center(self.modules[i])
                pts_x.append(cx)
                pts_y.append(cy)
            for tname in terminals:
                if tname in self.terminal_map:
                    tx, ty = self.terminal_map[tname]
                    pts_x.append(ca.MX(tx))
                    pts_y.append(ca.MX(ty))
            if len(pts_x) < 2:
                continue
            lse_x = (stable_logsumexp(pts_x, alpha) + stable_logsumexp([ -v for v in pts_x ], alpha)) / alpha
            lse_y = (stable_logsumexp(pts_y, alpha) + stable_logsumexp([ -v for v in pts_y ], alpha)) / alpha
            obj_terms.append((lse_x + lse_y) * (weight * self.wl_mult))

        if obj_terms:
            f = ca.sum1(ca.vertcat(*obj_terms))
        else:
            f = ca.MX(0)

        self.g_sym = ca.vertcat(*g) if g else ca.MX()
        self.f_sym = f

        self._lbg_list = lbg
        self._ubg_list = ubg

        self.nlp = {"x": self.x_sym, "p": self.params, "f": self.f_sym, "g": self.g_sym}
        self.solver = ca.nlpsol("solver", "ipopt", self.nlp, {
            "ipopt.print_level": 0 if not verbose else 5,
            "print_time": 0 if not verbose else 1,
            "ipopt.tol": otol,
            "ipopt.acceptable_tol": otol * 10,
            "ipopt.constr_viol_tol": rtol,
            "ipopt.acceptable_constr_viol_tol": rtol * 10,
            "ipopt.warm_start_init_point": "yes" if prev_vals is not None else "no",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.linear_solver": "mumps",
            "ipopt.max_iter": 1000,
            "ipopt.hessian_approximation": "limited-memory",
        })

    def solve(self, x0: list[float], current_tau_val: float, active_mask_vals: list[float], prev_vals: Optional[list[float]], step_cap_val: Optional[float], otol: float, rtol: float, verbose: bool = False) -> dict[str, Any]:
        # Rebuild solver instance for current iteration parameters and active set
        self._rebuild_solver_instance(current_tau_val, active_mask_vals, prev_vals, step_cap_val, otol, rtol, verbose)

        p_val = ca.vertcat(current_tau_val, ca.vertcat(*active_mask_vals), step_cap_val if step_cap_val is not None else 0.0)
        
        sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self._lbg_list, ubg=self._ubg_list, p=p_val)
        return {"x": sol["x"], "f": sol["f"], "g": sol.get("g", None)}

    @property
    def _lbg(self) -> list[float]:
        # this property is no longer used, bounds are returned from _rebuild_solver_instance
        return self._lbg_list

    @property
    def _ubg(self) -> list[float]:
        # this property is no longer used, bounds are returned from _rebuild_solver_instance
        return self._ubg_list

    # Dummy helper (kept for interface completeness)
    def _numeric(self, *_, **__) -> list[float]:
        return []


def visualize_iteration(
    ml: list[InputModule],
    og_names: list[str],
    terminal_map: dict[str, tuple[float, float]],
    die_width: float,
    die_height: float,
    iteration: int,
    hpwl: float,
    overlap: float,
    output_dir: str,
    tau: float
) -> None:
    """Visualize current layout with HPWL and overlap information"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  matplotlib not available, skipping visualization")
        return
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw die boundary
    ax.add_patch(patches.Rectangle(
        (0, 0), die_width, die_height,
        fill=False, edgecolor='darkblue', linewidth=3, linestyle='--'
    ))
    
    # Color palette for modules
    colors = plt.cm.tab20(range(20))
    color_idx = 0
    
    # Draw all modules (including rectilinear)
    for mod_idx, (trunk, Nb, Sb, Eb, Wb) in enumerate(ml):
        module_name = og_names[mod_idx] if mod_idx < len(og_names) else f"M{mod_idx}"
        color = colors[color_idx % len(colors)]
        color_idx += 1
        
        # Draw all rectangles of this module
        all_rects = [trunk] + Nb + Sb + Eb + Wb
        for rect_idx, (cx, cy, w, h) in enumerate(all_rects):
            x_left = cx - w/2
            y_bottom = cy - h/2
            
            rect = patches.Rectangle(
                (x_left, y_bottom), w, h,
                facecolor=color,
                edgecolor='black',
                linewidth=1.5 if rect_idx == 0 else 1.0,  # Thicker border for trunk
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add label only on trunk
            if rect_idx == 0 and w > 0.3 and h > 0.3:
                ax.text(cx, cy, module_name, 
                       ha='center', va='center', 
                       fontsize=8, weight='bold',
                       color='white' if sum(color[:3]) < 1.5 else 'black')
    
    # Draw terminals
    for term_name, (tx, ty) in terminal_map.items():
        ax.plot(tx, ty, 'ro', markersize=10, markeredgecolor='darkred', markeredgewidth=2)
        ax.text(tx, ty + 0.5, term_name, 
               ha='center', va='bottom',
               fontsize=7, color='red', weight='bold')
    
    # Set limits and aspect
    ax.set_xlim(-0.5, die_width + 0.5)
    ax.set_ylim(-0.5, die_height + 0.5)
    ax.set_aspect('equal')
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('X coordinate', fontsize=10)
    ax.set_ylabel('Y coordinate', fontsize=10)
    
    # Title with statistics
    ax.set_title(
        f'Iteration {iteration}\n'
        f'HPWL: {hpwl:.2f}, Overlap: {overlap:.4f}, τ: {tau:.2e}\n'
        f'Die: {die_width:.1f} × {die_height:.1f}, Modules: {len(ml)}, Terminals: {len(terminal_map)}',
        fontsize=12, pad=20
    )
    
    # Save figure
    output_file = os.path.join(output_dir, f"iter_{iteration:03d}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Plot saved: {output_file}")


def main(prog: Optional[str] = None, args: Optional[list[str]] = None) -> int:
    options = parse_options(prog, args)
    
    # Check if plotting is enabled
    plot_enabled = options.get("plot", False)
    if plot_enabled and not MATPLOTLIB_AVAILABLE:
        print("⚠️  Warning: --plot enabled but matplotlib not available")
        print("    Install matplotlib: pip install matplotlib")
        plot_enabled = False
    
    # Create plot directory if needed
    if plot_enabled:
        plot_dir = options.get("plot_dir", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        print(f"📁 Plot directory: {plot_dir}")
    
    (
        ml,
        al,
        xl,
        yl,
        wl,
        hl,
        die_width,
        die_height,
        hyper,
        max_ratio,
        og_names,
        terminal_map,
    ) = compute_options(options)

    last_x = None
    die_area = die_width * die_height
    overlap_threshold = die_area * 0.0085
    #overlap_threshold = 1e-5
    
    def compute_total_overlap(ml: list[InputModule]) -> float:
        """Compute total overlap area between all module pairs"""
        total_overlap = 0.0
        for mi in range(len(ml)):
            for mj in range(mi + 1, len(ml)):
                bi = ml[mi][0]  # trunk box: (cx, cy, w, h)
                bj = ml[mj][0]

                cx1, cy1, w1, h1 = bi
                cx2, cy2, w2, h2 = bj
                left1 = cx1 - 0.5 * w1
                right1 = cx1 + 0.5 * w1
                bottom1 = cy1 - 0.5 * h1
                top1 = cy1 + 0.5 * h1
                left2 = cx2 - 0.5 * w2
                right2 = cx2 + 0.5 * w2
                bottom2 = cy2 - 0.5 * h2
                top2 = cy2 + 0.5 * h2

                overlap_left = max(left1, left2)
                overlap_right = min(right1, right2)
                overlap_bottom = max(bottom1, bottom2)
                overlap_top = min(top1, top2)
                if overlap_left < overlap_right and overlap_bottom < overlap_top:
                    overlap_area = (overlap_right - overlap_left) * (overlap_top - overlap_bottom)
                    total_overlap += overlap_area
        return total_overlap
    
    for i in range(options["num_iter"]):

        base_tau = options.get("tau_initial", None)
        if base_tau is None:
            base_tau = 1e-3  
        tau_decay = options.get("tau_decay", 0.7)  
        
        current_tau = float(base_tau * (tau_decay ** float(i)))
        
        if options["verbose"]:
            print(f"Iteration {i+1}/{options['num_iter']}: tau = {current_tau:.6e}")

        max_dim = max(die_width, die_height)
        radius_val = float(options.get("radius", 1.0))
        radius_val = max(0.0, min(1.0, radius_val))
        dist_threshold = radius_val * max_dim

        def rect_from_box(b: BoxType) -> tuple[float, float, float, float]:
            cx, cy, w, h = b
            return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)

        def l1_gap(b1: BoxType, b2: BoxType) -> float:
            x1a, y1a, x2a, y2a = rect_from_box(b1)
            x1b, y1b, x2b, y2b = rect_from_box(b2)
            dx = max(0.0, max(x1a, x1b) - min(x2a, x2b))
            dy = max(0.0, max(y1a, y1b) - min(y2a, y2b))

            if dx == 0.0 and dy == 0.0:
                return 0.0
            return dx + dy

        active_mask_vals: list[float] = []
        pair_idx_counter = 0
        for mi in range(len(ml)):
            for mj in range(mi + 1, len(ml)):
                bi = ml[mi][0]  # trunk box
                bj = ml[mj][0]
                if l1_gap(bi, bj) <= dist_threshold:
                    active_mask_vals.append(1.0) # active
                else:
                    active_mask_vals.append(0.0) # inactive
                pair_idx_counter += 1

        # Compute trust-region step size: proportional to die dimension, gradually decreasing
        step_cap = 0.1 * max_dim * (1.0 - 0.6 * (i / max(1, options["num_iter"] - 1)))
        
        # Create CasadiLegalizer instance (rebuilt each iteration with updated ml)
        model = CasadiLegalizer(
            ml,
            al,
            xl,
            yl,
            wl,
            hl,
            die_width,
            die_height,
            hyper,
            max_ratio,
            og_names,
            wl_mult=options["wl_mult"],
            tau_initial=current_tau,  
            tau_decay=1.0,            
            otol_initial=options.get("otol_initial", 1e-1),
            otol_final=options.get("otol_final", 1e-4),
            rtol_initial=options.get("rtol_initial", 1e-1),
            rtol_final=options.get("rtol_final", 1e-4),
            tol_decay=options.get("tol_decay", 0.5),
            terminal_map=terminal_map,
        )
        

        current_otol = max(options.get("otol_initial", 1e-1) * (options.get("tol_decay", 0.5)**i), options.get("otol_final", 1e-4))
        current_rtol = max(options.get("rtol_initial", 1e-1) * (options.get("tol_decay", 0.5)**i), options.get("rtol_final", 1e-4))
        if options["verbose"]:
            print(f"    OTOL={current_otol:.6e}, RTOL={current_rtol:.6e}")


        if last_x is not None:
            model.x0 = list(map(float, last_x))

        # Solve optimization problem
        sol = model.solve(
            x0=model.x0,  # Use updated x0 (warm start if available)
            current_tau_val=current_tau,
            active_mask_vals=active_mask_vals,
            prev_vals=(list(map(float, last_x)) if last_x is not None else None),
            step_cap_val=step_cap if last_x is not None else None,
            otol=current_otol,
            rtol=current_rtol,
            verbose=options["verbose"],
        )
        x = sol["x"].full().flatten()
        fval = float(sol["f"]) if hasattr(sol["f"], "__float__") else float(sol["f"].full().item())
        

        prev_x = last_x  
        last_x = x
        
        # Extract solution and update ml for next iteration
        # Map solution back to ml structure (trunk, Nb, Sb, Eb, Wb)
        xi = 0
        new_ml: list[InputModule] = []
        for mod_idx, (trunk, Nb, Sb, Eb, Wb) in enumerate(ml):
            m = model.modules[mod_idx]
            
            # Extract all rectangles for this module
            new_rects: list[BoxType] = []
            for rect_idx in range(m.c):
                rx = float(x[xi]); ry = float(x[xi+1]); rw = float(x[xi+2]); rh = float(x[xi+3])
                xi += 4
                new_rects.append((rx, ry, rw, rh))
            
            # Reconstruct (trunk, Nb, Sb, Eb, Wb) structure
            new_trunk = new_rects[0]
            new_Nb = [new_rects[idx] for idx in m.N]
            new_Sb = [new_rects[idx] for idx in m.S]
            new_Eb = [new_rects[idx] for idx in m.E]
            new_Wb = [new_rects[idx] for idx in m.W]
            
            new_ml.append((new_trunk, new_Nb, new_Sb, new_Eb, new_Wb))


        ml = new_ml
        

        current_overlap = compute_total_overlap(ml)
        

        print(f"Iteration {i+1}/{options['num_iter']}: tau = {current_tau:.6e}, objective = {fval:.6f}, overlap = {current_overlap:.6f}")
        
        if options["verbose"]:
            print(f"    Overlap threshold = {overlap_threshold:.6f}")
        
        # Visualization if enabled
        if plot_enabled:
            visualize_iteration(
                ml=ml,
                og_names=og_names,
                terminal_map=terminal_map,
                die_width=die_width,
                die_height=die_height,
                iteration=i+1,
                hpwl=fval,
                overlap=current_overlap,
                output_dir=options.get("plot_dir", "plots"),
                tau=current_tau
            )
        
        # Iteration stop criterion: stop when overlap area < die_area * 0.0085
        if current_overlap < overlap_threshold:
            print(f"Early termination: overlap area {current_overlap:.6f} < threshold {overlap_threshold:.6f} (die_area * 0.0085)")
            break

    # Write output YAML with updated centers/sizes of ALL rectangles
    if options["outfile"] is not None and last_x is not None:
        net = Netlist(options["netlist"])
        xi = 0
        ml_iter = 0
        for m in net.modules:
            # Skip terminals (is_iopin), not just fixed
            if m.is_iopin:
                continue
            
            # Get corresponding structure from ml
            trunk, Nb, Sb, Eb, Wb = ml[ml_iter]
            ml_iter += 1

            # Update all rectangles for this module
            if m.rectangles:
                # Update trunk (rectangles[0])
                tx = float(last_x[xi]); ty = float(last_x[xi+1]); tw = float(last_x[xi+2]); th = float(last_x[xi+3])
                xi += 4
                m.rectangles[0].center.x = tx
                m.rectangles[0].center.y = ty
                m.rectangles[0].shape.w = tw
                m.rectangles[0].shape.h = th

                # Update branches in order (Nb + Sb + Eb + Wb)
                k = 1  # Next rectangle index
                for group in (Nb, Sb, Eb, Wb):
                    for _ in group:
                        if k < len(m.rectangles):
                            gx = float(last_x[xi]); gy = float(last_x[xi+1]); gw = float(last_x[xi+2]); gh = float(last_x[xi+3])
                            xi += 4
                            m.rectangles[k].center.x = gx
                            m.rectangles[k].center.y = gy
                            m.rectangles[k].shape.w = gw
                            m.rectangles[k].shape.h = gh
                            k += 1
                        else:
                            # Consume variable indices to avoid misalignment
                            xi += 4
            else:
                # No rectangles, but still consume variable indices
                xi += 4  # trunk
                xi += 4 * (len(Nb) + len(Sb) + len(Eb) + len(Wb))

        net.write_yaml(options["outfile"])

    return 0


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    sys.exit(main())


