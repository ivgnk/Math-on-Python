'''
example from
https://pypi.org/project/sympy-slider/
'''

import numpy as np
from sympy import sin, cos
from sympy.abc import a, b, x

from sympy_slider import slider

expr = a*cos(x) + b*sin(x)

params = slider(
    expr,
    x,
    params=dict(
        a=(0, -1, 1),
        b=(0, -1, 1),
    ),
    xdata=np.linspace(0, 2*np.pi, 1000),
    lambdify_kws=dict(
        modules='numpy',
    ),
    plot_kws=dict(
        color='blue',
        linestyle='--',
    ),
    use_latex=True,
)