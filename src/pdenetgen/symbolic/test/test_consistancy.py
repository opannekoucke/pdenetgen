from unittest import TestCase
from pdenetgen.symbolic import remove_eval_derivative
from pdenetgen.symbolic.finite_difference import finite_difference
from sympy import Function, Derivative, symbols
class TestFiniteDifferenceConsistancy(TestCase):
    """
    Test of the **second-order consistancy** of finite difference operator

    """

    consistancy_order = 2 # theoretical consistancy order for the package.

    def test_even_order_derivative_consistancy(self):            
        # Set function
        x, dx = symbols('x dx')
        u = Function('u')(x)
        
        # Set derivative of even order
        expr = Derivative(u,x,4)

        # Compute the finite difference 
        fd_expr = finite_difference(expr)

        # Validation of the consistancy
        consistance_fd_expr = fd_expr.series(dx,0, self.consistancy_order)
        consistance_fd_expr = consistance_fd_expr.removeO()
        consistance_fd_expr = remove_eval_derivative(consistance_fd_expr)
        self.assertEqual(consistance_fd_expr, expr)        

    def test_odd_order_derivative_consistancy(self):        
        # Set function
        x, dx = symbols('x dx')
        u = Function('u')(x)
        
        # Set derivative of even order
        expr = Derivative(u,x,5)

        # Compute the finite difference 
        fd_expr = finite_difference(expr)

        # Validation of the consistancy
        consistance_fd_expr = fd_expr.series(dx,0, self.consistancy_order)
        consistance_fd_expr = consistance_fd_expr.removeO()
        consistance_fd_expr = remove_eval_derivative(consistance_fd_expr)
        self.assertEqual(consistance_fd_expr, expr)


if __name__ == '__main__':
    import unittest
    unittest.main()