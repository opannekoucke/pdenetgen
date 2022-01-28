from unittest import TestCase
from pdenetgen.symbolic import Eq
from sympy import symbols

class TestEq(TestCase):
    """
    Test of the expectation operator E()
    This operator should be:

    * Linear (for non-random components)
    * Idempotent

    """

    def test_elementary(self):
        a,b,c,d = symbols('a b c d')
        eq1 = Eq(a,b)
        eq2 = Eq(c,d)
        self.assertEqual(eq1+eq2, Eq(*[eq1.args[k]+eq2.args[k] for k in range(2)]))
        self.assertEqual(eq1-eq2, Eq(*[eq1.args[k]-eq2.args[k] for k in range(2)]))
        self.assertEqual(eq1*2, Eq(*[2*eq1.args[k] for k in range(2)]))




