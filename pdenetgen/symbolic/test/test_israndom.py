from unittest import TestCase
from pydap.symbolic.random import Expectation, omega, israndom, IncoherentEquality
from pydap.symbolic import Eq
from sympy import Function, Derivative, symbols, Integer

x, t, nu, kappa = symbols('x t \\nu \\kappa')
u = Function('u')(x, t, omega)
f = Function('f')

class TestExpectation(TestCase):
    """
    Test of the expectation operator E()
    This operator should be:

    * Linear (for non-random components)
    * Idempotent

    """

    def test_basic_operator(self):
        # Validation Mul
        self.assertTrue(israndom(u*x))
        self.assertFalse(israndom(x*x))
        self.assertTrue(israndom(u*Derivative(u,x)))

    def test_basic(self):
        self.assertTrue( israndom(omega) )
        self.assertTrue( israndom(u) )
        self.assertFalse( israndom(x) )
        self.assertFalse( israndom(Integer(2)) )
        self.assertFalse( israndom(Expectation(u)) )
        self.assertFalse( israndom(Derivative(Expectation(u), x)) )
        self.assertFalse( israndom(Eq(Expectation(u), t)) )
        self.assertFalse( israndom(f(Expectation(u))) )
        try:
            israndom(Eq(Derivative(u, t), x))
        except IncoherentEquality:
            'Incoherence has been found'
            pass
        """
        assert israndom(u * Derivative(u, x))
        assert israndom(Eq(Derivative(u, t), Derivative(u, x)))
        """
