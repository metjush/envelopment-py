# Data Envelopment Analysis

This repository implements the DEA (Data Envelopment Analysis) algorithm in Python, based on `numpy` (`scipy` stack).
The model is built on the specifications and formulas described and derived in Sherman and Zhu (2006, pp. 49-89).

## Brief description
The model is estimated by minimizing the `theta` expression for each unit, where theta is `uy/vx`, subject to constraints as defined in
equation (2.2) in Sherman and Zhu. This dual linear program is optimized using *Sequential Least Squares Programming*, as implemented in `scipy` with the 
`scipy.optimize.fmin_slsqp` method. 

The model is optimized unit-by-unit, with weights being initialized randomly, with a uniform distribution between `-0.5` and `0.5`. 
After each unit's weights are optimized and the algorithm converges, `theta` (efficiency) of that unit is calculated and saved.

After all units are fit, the `fit()` function prints `thetas` for each unit.

## Sources
Sherman and Zhu (2006) *Service Productivity Management*, Improving Service Performance using Data Envelopment Analysis (DEA) [Chapter 2]
ISBN: 978-0-387-33211-6
