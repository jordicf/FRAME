# Normalize grid into rectangles

## Problem Statement

Given a fuzzy (non-necessarily uniform) grid of rectangles, each one with a particular density for every module, and a particular module to normalize, finds the set of rectangles that best conform to that shape. The definition of "best conforms" depends on the parameters set.

## Installation

**Disclaimer:** This will not be the final method of installation.

Run compile.sh to compile the greedy component. The program can now be run by:

```
python main.py
```

## Usage
```
Usage:
 python main.py [file name] [options]

Options:
 --minarea : Minimizes the total area while guaranteeing a minimum coverage of the original
 --maxdiff : Maximizes the difference between the inner area and the outer area
 --minerr  : Minimizes the error (Default)
 --sf [d]  : Manually set the factor to number d (Not recommended)
```

### Minarea option

**Maximize** $(\sum A_i p_i x_i) / (\sum A_i x_i)$

**Subject To** $\sum A_i x_i \geq 0.89 \sum A_i p_i$

### Maxdiff option

**Maximize** $3\sum A_i p_i x_i - \sum A_i x_i$

### Minerr option

**Maximize** $2\sum A_i p_i x_i - \sum A_i x_i$

### sf option

if d < 1 then:
.  **Maximize**  $(\sum A_i p_i x_i) / (\sum A_i x_i)$
.  **Subject To** $\sum A_i x_i \geq d \sum A_i p_i$
else:
.  **Maximize** $d\sum A_i p_i x_i - \sum A_i x_i$


