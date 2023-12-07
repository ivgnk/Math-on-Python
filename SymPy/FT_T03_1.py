'''
For Field Theory lessons
Task 03_1:  вычисления и  построение графиков
https://docs.sympy.org/latest/modules/plotting.html
'''
# class sympy.plotting.plot.Plot(*args, title=None, xlabel=None, ylabel=None, zlabel=None, aspect_ratio='auto',
# xlim=None, ylim=None, axis_center='auto', axis=True, xscale='linear', yscale='linear', legend=False, autoscale=True,
# margin=0, annotations=None, markers=None, rectangles=None, fill=None, backend='default', size=None, **kwargs)[source]

import sympy as sym
from sympy.plotting import plot
import spb # sympy-plot-backends

x = sym.Symbol('x')
y = sym.Symbol('y')

print('\n--- (1) --- Дифференцирование')
f1 = sym.sin(x)
f2 = sym.diff(sym.sin(x), x)
print(f'{sym.diff(sym.sin(x), x)=}')
plot( f1, f2, title = 'diff' , xlabel=' x ', ylabel=' y ' , legend = True, axis_center = (-12, -1.1) )

print('\nвычисление производные более высоких порядков')
f1 = sym.diff(sym.sin(2 * x), x, 1)
print(f'{sym.diff(sym.sin(2 * x), x, 1) = }')  # 2*cos(2*x)
f2 = sym.diff(sym.sin(2 * x), x, 2)
print(f'{sym.diff(sym.sin(2 * x), x, 2) = }')  # -4*sin(2*x)
f3 = sym.diff(sym.sin(2 * x), x, 3)
print(f'{sym.diff(sym.sin(2 * x), x, 3) = }')  # -8*cos(2*x)

plot( f1, f2, f3, title = 'High order derivatives' , xlabel=' x ', ylabel=' y ' , legend = True, axis_center = (-12, -1.1) ) #
# Manually set labels for each line in the legend

print('\n--- (2) --- Вторая библиотека для построения графиков SymPy')
# https://pypi.org/project/sympy-plot-backends
# https://sympy-plot-backends.readthedocs.io/en/latest
# https://sympy-plot-backends.readthedocs.io/en/latest/overview.html#plotting-functions
# https://stackforgeeks.com/blog/how-can-i-manually-specify-text-for-legends-in-sympyplottingplot

spb.plot( f1, f2, f3, title = 'High order derivatives' , xlabel=' x ', ylabel=' y ' , legend = True,
          label = ['Fun1','Fun2','Fun3'],axis_center = (-12, -1.1) )