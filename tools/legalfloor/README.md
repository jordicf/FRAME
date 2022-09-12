# Legalfloor: Floorplan Legalization Tool

## Problem Statement

Given a (non-necessarily legal) floorplan, legalize the floorplan (aka, make no two rectangles overlap) and minimize the 
wirelength of the connections hypergraph as much as possible.

<img src="../../doc/pict/legalfloor_example.png" alt="Grid normalization problem statement" style="height: 360px;"/>

## Usage

```
Usage: legalfloor.py [options]

A tool for module legalization

positional arguments:
  netlist               Input netlist (.yaml)
  die                   Input die (.yaml)

options:
  -h, --help            show this help message and exit
  --max_ratio MAX_RATIO The maximum allowable ratio for a rectangle
  --plot                Plots the problem together with the solutions found
  --verbose             Shows additional debug information```
```
