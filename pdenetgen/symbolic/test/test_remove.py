import unittest
from pydap.symbolic import remove_eval_derivative
from sympy import symbols, Function, Derivative


class MyTestCase(unittest.TestCase):
    def test_remove_eval_derivative(self):
        t, x, dx = symbols('t x dx')
        u = Function('u')(t, x)
        exemple = u.subs(x, x + dx).series(dx, 0, 3).removeO()
        self.assertEqual(remove_eval_derivative(exemple).coeff(dx), Derivative(u, x))


if __name__ == '__main__':
    unittest.main()
