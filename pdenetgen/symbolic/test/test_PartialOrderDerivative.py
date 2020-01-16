import unittest
from sympy import symbols
from pydap.symbolic.pkfcas import PartialOrderDerivative

class MyTestCase(unittest.TestCase):
    coords = symbols('x y')
    x,y = coords

    def test_entrence_format(self):
        alpha = PartialOrderDerivative(self.coords, (5,0))
        self.assertEqual(alpha, PartialOrderDerivative(self.coords, alpha.as_sequence))
        self.assertEqual(alpha, PartialOrderDerivative(self.coords, alpha.as_couples))

    def test_type_error(self):
        # Test type error:
        try:
            PartialOrderDerivative(self.coords,('x',1))
        except TypeError:
            test=True # Ok bien vu
        except:
            test = False
        self.assertTrue(test)

if __name__ == '__main__':
    unittest.main()
