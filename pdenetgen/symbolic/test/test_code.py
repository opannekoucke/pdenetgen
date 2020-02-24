import unittest

from pdenetgen.symbolic.finite_difference import Code
from sympy import Function, symbols

class TestCode(unittest.TestCase):

    t, x, y = symbols('t x y')
    dt, dx, dy = symbols('dt dx dy')

    def test_code_time(self):
        f = Function('f')(self.t+self.dt,self.x, self.y)
        self.assertEqual(Code(f).code, 'fp1')
        f = Function('f')(self.t -2* self.dt, self.x, self.y)
        self.assertEqual(Code(f).code, 'fm2')

    def test_code_index(self):
        f = Function('f')(self.t,self.x+self.dx, self.y)
        self.assertEqual(Code(f).code, "f[np.ix_(self.index('x',1),self.index('y',0))]")
        f = Function('f')(self.t+self.dt,self.x+self.dx, self.y)
        self.assertEqual(Code(f).code, "fp1[np.ix_(self.index('x',1),self.index('y',0))]")


if __name__ == '__main__':
    unittest.main()
