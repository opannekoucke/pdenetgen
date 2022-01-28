import unittest
from pdenetgen.symbolic import CoordinateSystem
from sympy import symbols, Function, Matrix, Derivative

class TestCoordinateSystem(unittest.TestCase):

    coords = CoordinateSystem(symbols('x y'))

    def test_compatibility(self):
        # Case not compatible
        f = Function('f')(*symbols('t x'))
        self.assertTrue(not self.coords.is_compatible(f))

        # Case compatible
        f = Function('f')(*symbols('t x y'))
        self.assertTrue(self.coords.is_compatible(f))

    def test_gradient_div(self):
        # Case compatible
        f = Function('f')(*symbols('t x y'))
        x, y = symbols('x y')

        # Computation of the gradient
        grad_f = self.coords.gradient(f)
        self.assertEqual(grad_f,Matrix([Derivative(f,x), Derivative(f,y)]))

        # Computation of the divergent of the gradient which should be the laplacian here
        div_grad_f = self.coords.div(grad_f)
        lap_f = Derivative(f,x,2)+Derivative(f,y,2)
        self.assertEqual(div_grad_f, lap_f)

if __name__ == '__main__':
    unittest.main()
