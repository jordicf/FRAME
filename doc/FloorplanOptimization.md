# Numerical Optimization Floorplanning

## Problem Statement

Given a set of modules, each one with a set of rectangles, and a hypergraph connecting all the modules, define a mathematical optimization model that minimizes the wire length of the modules while keeping the modules restrained inside of a fixed area.

## Variables and Constants

### Constants

We have a set of modules $M$.

Every module $m \in M$ has a set of rectangles $R(m) = \{0\} \cup R_N(m) \cup R_S(m) \cup R_E(m) \cup R_W(m)$, where $0$ is the ''troncal" rectangle, $R_N(m)$ are the rectangles attached to the troncal rectangle by the north, $R_S(m)$ are the rectangles attached to the troncal rectangle by the south, and so on. These $R_N(m)$, $R_S(m)$... sets are disjoint, and do not contain $0$.

Every module $m \in M$ also has a set minimal area $A_m > 0$.

**Hard** modules $m$ also have, per every rectangle $i \in R(m)$, a fixed width $W_{im}$, a fixed height $H_{im}$, and per every rectangle that's not the troncal one $i \in R(m) \setminus \{0\}$, a relative position $X_{im}$ and $Y_{im}$ to the troncal rectangle.

**Fixed** modules $m$ have, in addition to everything stated previously, $X_{0m}$ and $Y_{0m}$ the coordinates of the troncal rectangle.

Let $\Omega$ be the hypergraph, and let $\langle \omega, S \rangle \in \Omega$ be an edge on the hypergraph, where $\omega$ is the weight of the edge, or "how important" it is to keep this edge short, and let $S$ be a multisubset of $M$ that represents all the modules that are connected by this edge.

### Variables

For every module $m$, every rectangle $i \in R(m)$ has the following real-valued variables:

<ul>
  <li>$x_{im}$: x-coordinate of the center of the rectangle.</li>
  <li>$y_{im}$: y-coordinate of the center of the rectangle.</li>
  <li>$w_{im}$: width of the rectangle.</li>
  <li>$h_{im}$: height of the rectangle.</li>
</ul>

Also, every module $m$ has the following variables / expressions:
<ul>
  <li>$x_{m} = \frac{\sum_{i \in R(m)} x_{im} w_{im} h_{im}}{\sum_{i \in R(m)} w_{im} h_{im}}$ the x coordinate of the center of mass of module $m$.
  <li>$y_{m} = \frac{\sum_{i \in R(m)} y_{im} w_{im} h_{im}}{\sum_{i \in R(m)} w_{im} h_{im}}$ the y coordinate of the center of mass of module $m$.
</ul>

## Minimal Area Requirements

$\text{for every module }m:$

$\hspace{20px}\sum_{i \in R(m)} w_{im} * h_{im} \geq A_m$

## Keep Modules Attached

$\text{for every }\textbf{non-hard (soft)}\text{ module }m:$

$\hspace{20px}\text{for every rectangle }i \in R_N(m):$

$\hspace{40px} y_{im} = y_{0m} + \frac{1}{2} (h_{0m} + h_{im})$

$\hspace{40px} x_{im} \geq x_{0m} - \frac{1}{2} (w_{0m} - w_{im})$

$\hspace{40px} x_{im} \leq x_{0m} + \frac{1}{2} (w_{0m} - w_{im})$

$\hspace{20px}\text{for every rectangle }i \in R_S(m):$

$\hspace{40px} y_{im} = y_{0m} - \frac{1}{2} (h_{0m} + h_{im})$

$\hspace{40px} x_{im} \geq x_{0m} - \frac{1}{2} (w_{0m} - w_{im})$

$\hspace{40px} x_{im} \leq x_{0m} + \frac{1}{2} (w_{0m} - w_{im})$

$\hspace{20px}\text{for every rectangle }i \in R_E(m):$

$\hspace{40px} x_{im} = x_{0m} + \frac{1}{2} (w_{0m} + w_{im})$

$\hspace{40px} y_{im} \geq y_{0m} - \frac{1}{2} (h_{0m} - h_{im})$

$\hspace{40px} y_{im} \leq y_{0m} + \frac{1}{2} (h_{0m} - h_{im})$

$\hspace{20px}\text{for every rectangle }i \in R_W(m):$

$\hspace{40px} x_{im} = x_{0m} - \frac{1}{2} (w_{0m} + w_{im})$

$\hspace{40px} y_{im} \geq y_{0m} - \frac{1}{2} (h_{0m} - h_{im})$

$\hspace{40px} y_{im} \leq y_{0m} + \frac{1}{2} (h_{0m} - h_{im})$

## Soft, Hard and Fixed Modules

Non-hard (aka. soft) modules do not require any additional constraints.

$\text{for every }\textbf{hard}\text{ module }m:$

$\hspace{20px}\text{for every rectangle }i \in R(m):$

$\hspace{40px}w_{im} = W_{im}$

$\hspace{40px}h_{im} = H_{im}$

$\hspace{40px}\text{if }i \neq 0$

$\hspace{60px}x_{im} = x_{0m} + X_{im}$

$\hspace{60px}y_{im} = y_{0m} + Y_{im}$

$\text{for every }\textbf{fixed}\text{ module }m:$

$\hspace{20px}x_{0m} = X_{0m}$

$\hspace{20px}y_{0m} = Y_{0m}$



## No Overlaps

### No Intra-Module Overlap

$\text{for every }\textbf{non-hard (soft)}\text{ module }m:$

$\hspace{20px}\text{for every two consecutive rectangles }i, j \in R_N(m)\text{ s.t. }x_{im} < x_{jm}:$

$\hspace{40px} x_{im} + \frac{1}{2}w_{im} \leq x_{jm} - \frac{1}{2} w_{jm}$

$\hspace{20px}\text{for every two consecutive rectangles }i, j \in R_S(m)\text{ s.t. }x_{im} < x_{jm}:$

$\hspace{40px} x_{im} + \frac{1}{2}w_{im} \leq x_{jm} - \frac{1}{2} w_{jm}$

$\hspace{20px}\text{for every two consecutive rectangles }i, j \in R_E(m)\text{ s.t. }y_{im} < y_{jm}:$

$\hspace{40px} y_{im} + \frac{1}{2}h_{im} \leq y_{jm} - \frac{1}{2} h_{jm}$

$\hspace{20px}\text{for every two consecutive rectangles }i, j \in R_W(m)\text{ s.t. }y_{im} < y_{jm}:$

$\hspace{40px} y_{im} + \frac{1}{2}h_{im} \leq y_{jm} - \frac{1}{2} h_{jm}$

### No Inter-Module Overlap

$\text{for every two modules }m, n:$

$\hspace{20px}\text{for every two rectangles }i \in R(m), j \in R(n):$

$\hspace{40px}M( (x_{im} - x_{jm})^2 - \frac{1}{4} (w_{im} + w_{jm}), (y_{im} - y_{jm})^2 - \frac{1}{4} (h_{im} + h_{jm}) ) \geq 0$

$\hspace{40px}\text{Where }M(x, y) = \frac{1}{2} (x + y + \sqrt{ (x - y)^2 + 4 \tau^2 } ) \text{ for some small } \tau > 0$



## Keep Everything Inside the Working Area
$\text{for every module }m\text{, rectangle }i \in R(m):$

$\hspace{20px}x_{im} - \frac{1}{2}w_{im} >= X_{low}$

$\hspace{20px}x_{im} + \frac{1}{2}w_{im} <= X_{high}$

$\hspace{20px}y_{im} - \frac{1}{2}h_{im} >= Y_{low}$

$\hspace{20px}y_{im} + \frac{1}{2}h_{im} <= Y_{high}$

## Other Constraints

### Variable domains
$\text{for every module }m\text{, rectangle }i \in R(m):$

$\hspace{20px}w_{im} > 0$

$\hspace{20px}h_{im} > 0$


### Avoid Thin Rectangles

#### Approach 1: Avoid Extreme Proportions

Let $f(x) = x + \frac{1}{x} - 1$

This function serves as a metric of, given a ratio $x = \frac{w}{h}$, how "thin" the rectangle is. If $x = 1$ ($w = h$), this function returns $1$, and returns a number $> 1$ otherwise. The more extreme the ratio is, the greater the number it returns.


$\text{for every module }m\text{, rectangle }i \in R(m):$

$\hspace{20px}f(\frac{w_{im}}{h_{im}}) \leq f(\rho)$

Where $\rho$ is some maximal allowable ratio.

#### Approach 2: Set a Minimum Value to Width and Height

$\text{for every module }m\text{, rectangle }i \in R(m):$

$\hspace{20px}w_{im} \geq W_{min}$

$\hspace{20px}h_{im} \geq H_{min}$

Where $W_{min}$ and $H_{min}$ are, quite intuitively, a minimum value for the width and height of a rectangle.

## Objective function

### Complete Graph Metric

Minimize $\sum_{\langle \omega, S \rangle \in \Omega} \sum_{m, n \in S} \omega^2 \cdot ( (x_{m} - x_{n})^2 + (y_{m} - y_{n})^2 )$

### Star Graph Metric

For every multisubset $S$ of $M$, we define the center as:

$x(S) = \frac{1}{|S|} \sum_{m \in S} x_m$

$y(S) = \frac{1}{|S|} \sum_{m \in S} y_m$

Minimize $\sum_{\langle \omega, S \rangle \in \Omega} \sum_{m \in S} 4\omega^2 \cdot ( (x_{m} - x(S))^2 + (y_{m} - y(S))^2 )$
