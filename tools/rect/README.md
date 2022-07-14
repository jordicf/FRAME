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

When running the minarea option, the program tries to maximize the density of the enclosed area all while ensuring the total area is at least a percentage of the original area. It's called minarea because, in practice, this option finds the minimum area with the maximum density.

**Maximize** $(\sum A_i p_i x_i) / (\sum A_i x_i)$

**Subject To** $\sum A_i x_i \geq 0.89 \sum A_i p_i$

### Maxdiff option

When running the maxdiff option, the program maximizes the difference between the area inside and the area outside of the rectangle.

The area inside the rectangles can be written as $\sum A_i p_i x_i$, the area outside of the area is $\sum A_i p_i (1 - x_i)$, and the area inside the rectangles that's not part of the module is $\sum A_i (1 - p_i) x_i$. The difference is:

**Maximize** $\sum A_i p_i x_i - \sum A_i p_i (1 - x_i) - \sum A_i (1 - p_i) x_i$

Simplifying down the formula, we get:

**Maximize** $3\sum A_i p_i x_i - \sum A_i x_i - \sum A_i p_i$

Note how $\sum A_i p_i$ is just a constant (does not depend on any variable of the model) and therefore we can simplify it down to:

**Maximize** $3\sum A_i p_i x_i - \sum A_i x_i$

### Minerr option

When running the minerr option, the program minimizes the error. The error is just the last two terms in the original formulation of the last option. Simplifying:

**Maximize** $2\sum A_i p_i x_i - \sum A_i x_i$

### sf option

The sf option is a generalitazion of the other two options.

if d < 1 then:

.  **Maximize**  $(\sum A_i p_i x_i) / (\sum A_i x_i)$

.  **Subject To** $\sum A_i x_i \geq d \sum A_i p_i$

else:

.  **Maximize** $d\sum A_i p_i x_i - \sum A_i x_i$

If $d < 1$, the option runs just as the minarea option, except the arbitrary 0.89 constant is changed to whatever value you set for $d$.

If $d \geq 1$, the option runs just as the maxdiff or the minerr options, but changing the 3 or 2 constant respectively with $d$.

This option was added for testing purposes, and is not really recomended.
