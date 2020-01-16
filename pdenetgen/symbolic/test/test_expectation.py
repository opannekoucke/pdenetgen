from unittest import TestCase
from pydap.symbolic.random import Expectation, omega, israndom
from sympy import Function, Derivative, symbols, I, latex


class TestExpectation(TestCase):
    """
    Test of the expectation operator E()
    This operator should be:

    * Linear (for non-random components)
    * Idempotent

    """

    def test_print(self):
        X = Function('X')(omega)
        # Validation du type
        self.assertEqual(type(Expectation(X)), Expectation )
        # Validation latex
        self.assertEqual(latex(Expectation(X)), '{\\mathbb E}\\left(X{\\left(\\omega \\right)}\\right)')

    def test_elementary(self):
        x, t, omega = symbols('x t omega')
        X = Function('X')(x, t, omega)
        dX = Derivative(X, x)
        f = Function('f')(x, t)
        self.assertEqual(Expectation(X * dX * dX), Expectation(X * dX ** 2))

    def test_eval(self):
        x, t, omega = symbols('x t omega')
        X = Function('X')(x, t, omega)
        dX = Derivative(X, x)
        f = Function('f')(x, t)
        self.assertEqual(Expectation(f * X), f * Expectation(X))
        self.assertEqual(Expectation(5. * X), 5. * Expectation(X))
        self.assertEqual(Expectation((1 + I * 5) * X),  Expectation(X)+5*I*Expectation(X))
        self.assertEqual(Expectation(X + X), 2 * Expectation(X))
        self.assertEqual(Expectation(X * X), Expectation(X ** 2))
        self.assertEqual(Expectation(X * dX), Expectation(X * dX))
        self.assertEqual(Expectation(2 * X * dX), 2 * Expectation(X * dX))
        # b) Test of addition
        self.assertEqual(Expectation(X + f * X), Expectation(X) + Expectation(f * X))

    def test_idempotence(self):
        x = symbols('x')
        X = Function('X')(omega)
        Y = Function('Y')(x, omega)
        self.assertEqual(Expectation(X).has(omega), False)
        self.assertEqual(Expectation(Y).has(x), True)
        self.assertEqual(Expectation(Expectation(X)), Expectation(X))

    def test_with_israndom(self):
        t = symbols('t')
        X = Function('X')(t,omega)
        self.assertFalse(israndom(Derivative(Expectation(X),t)))

