"""
Contains often used dynamics given by a system of PDE id 1D, 2D
"""

from sympy import symbols, Derivative, Function
from .util import PDESystem, Eq, CoordinateSystem, Matrix

# 1D equations

t,x,y = symbols('t x y')
kappa, eta = symbols('\\kappa \\eta')

u = Function('u')(t,x)

burgers = PDESystem(
        Eq(Derivative(u,t), -u*Derivative(u,x)+kappa*Derivative(u,x,2)),
        )

kuramoto = PDESystem(
        Eq(Derivative(u,t), -u*Derivative(u,x)-kappa*Derivative(u,x,2)-eta*Derivative(u,x,4)),
        )

diffusion = PDESystem(
        Eq(Derivative(u,t), kappa*Derivative(u,x,2)),
        )


c = Function('u')(t,x)
u = Function('u')(x)

advection = PDESystem(
        Eq(Derivative(c,t), -u*Derivative(c,x))
)

# 2D equations

coords = CoordinateSystem((x,y))

c = Function('c')(t,x,y)
u = Function('u')(x,y)
v = Function('v')(x,y)

advection_in_2D = PDESystem(
        Eq(Derivative(c,t), -u*Derivative(c,x)-v*Derivative(c,y))
)


kappa11 = Function('\\kappa_{11}')(x,y)
kappa12 = Function('\\kappa_{12}')(x,y)
kappa22 = Function('\\kappa_{22}')(x,y)
kappa_tensor = Matrix([[kappa11,kappa12],[kappa12,kappa22]])
u = Function('u')(t, x, y)

diffusion_in_2D = PDESystem(
                Eq(Derivative(u,t), coords.div(kappa_tensor*coords.gradient(u)).doit())
)


