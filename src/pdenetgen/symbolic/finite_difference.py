from .util import Eq, get_function_coordinates, get_total_order
import warnings

def get_subs_derivative(derivative):
    """
    Compute the finite difference of a derivative on a regular grid at the second order of consistancy.
    
    Can compute the derivative of u(t,x,y+dy) in x 
    but not the derivative of u(t,x+dx,y+dy) !!!!
    
    Description
    -----------
    
    consistancy at second order means:    
    finite_difference_version - derivative = O(||dx||^2)
    """
    
    """
    Code explaination
    -----------------
    
    We illustrate the code on the following example:
        Derivative(u,(x,2),(y,1))
        
    The derivative is replaced by 
        Derivative( Derivative(u,(x,2)) , (y,1))
    
    Then, the computation follows the steps:
    1. the finite difference approximation replaces the internal derivative, Derivative(u,(x,2)),
        by its finite difference approximation. 
        
        finite_diff = Finite_Diff_of_(Derivative(u,x,2))
        
    2. then the approximation is 
    
        subs_derivative = Derivative(finite_diff, y,1)

    """
    from sympy import Derivative, Integer, symbols
    from numpy import arange

    # Get arguments from the derivative
    func_part = derivative.args[0]
    xi, oi = derivative.args[1]
    dxi = symbols('d' + str(xi))    
    
    # 1. Compute the finite difference of the derivation with respect to the first argument
    
    # 1.1 set points of the regular grid where the computation should be made.    
    # These points are selected to ensure a **second order consistency** of the finite difference approximation
    # --> modify the points to increase the consistency. 
    if oi%2==1:        
        points = [xi+Integer(k)*dxi for k in arange(-oi,oi+1) if k%2==1]                    
    else:
        #
        # For even derivative, this is equivalent to :
        #   derivative.as_finite_difference(dxi, wrt=xi)
        #
        points = [xi+Integer(k)*dxi for k in arange(-oi//2,oi//2+1)]

    # 1.2 Substitution of the selected part of the derivative: Derivative(.. , xi,oi)
    subs_derivative = Derivative(func_part,xi,oi).as_finite_difference(points)

    # 2. Computation of the remaining partial differential (if needed)
    if len(derivative.args)>2:
        subs_derivative = Derivative(subs_derivative,*derivative.args[2:]).doit().expand()        

    return subs_derivative

def finite_difference_derivative(derivative):
    """ Finite differenciate a single derivative

    Documentation
    -------------

        from sympy, see: Fornberg1988MC

    Example:
        >>> finite_difference(Derivative(f(x),x))
        f(x+dx)/2*dx - f(x-dx)/2*dx
    """
    # todo: eliminate the option `regular_grid` -- the computation is now always on a regular grid !!!!
    from sympy import Derivative

    expr = derivative.doit()

    while True:
        # -1- Find all derivatives in the expression
        derivatives = expr.atoms(Derivative)
        if derivatives == set():
            # if no derivative then exit loop !
            break
        # -2- Replace all derivatives found above
        for derivative in derivatives:
            subs_derivative = get_subs_derivative(derivative)
            expr = expr.subs(derivative, subs_derivative).expand()

    return expr

def finite_difference(expr):
    """ Finite differenciate derivative of an expression on **a regular grid**

    Documentation
    -------------

        from sympy, see: Fornberg1988MC

        Note that the default sympy computation of finite-difference approximation implies intermediate points
        that are not on the regular grid e.g. the finite difference of Derivative(u,x,3) implies point as x+3dx/2,
        which are not on the grid.

    Example
    -------

        >>> finite_difference(Derivative(f(x),x))
        f(x+dx)/2*dx - f(x-dx)/2*dx
    """
    # todo: eliminate the option `regular_grid` -- the computation is now always on a regular grid !!!!
    from sympy import Derivative

    expr = expr.doit()

    # -1- Find all derivatives in the expression
    derivatives = expr.atoms(Derivative)

    # -2- Ordering of the derivative by total orders
    derivatives_of_order = {}
    for derivative in derivatives:
        # i. Compute the order
        total_order = get_total_order(derivative)

        # ii. Set the dic for new order
        if total_order not in derivatives_of_order:
            derivatives_of_order[total_order] = []

            # iii. Add the derivative at the appropriate order
        derivatives_of_order[total_order].append(derivative)

    # -3- Substitute derivative by decreasing orders
    #     e.g. $\partial_x^2 u$ is replaced before $\partial_x u$ so that the finite diff. of $\partial_x u$ is not
    #     substitute in $\partial_x^2 u$

    # 3.1 Compute the decrease ordering of derivatives
    decreasing_orders = list(derivatives_of_order.keys())
    decreasing_orders.sort(reverse=True)

    # 3.2 Replace derivative by decreasing orders
    for order in decreasing_orders:
        for derivative in derivatives_of_order[order]:
            subs_derivative = finite_difference_derivative(derivative)
            expr = expr.subs(derivative, subs_derivative).expand()

    return expr

def _sympy_finite_difference(expr):
    """ Compute the finite difference of an expression using the raw second order approximation of sympy.

    Documentation
    -------------

        from sympy, see: Fornberg1988MC

    Example:
        >>> sympy_finite_difference(Derivative(f(x),x))
        f(x+dx)/2*dx - f(x-dx)/2*dx

    """
    from sympy import Derivative, symbols

    expr = expr.doit()

    while True:
        # -1- Find all derivatives in the expression
        derivatives = expr.atoms(Derivative)
        if derivatives == set():
            # if no derivative then exit loop !
            break
        # -2- Replace all derivatives found above
        for derivative in derivatives:
            # a) get first 'wrt' variable
            xi = derivative.args[1][0]
            dxi = symbols('d' + str(xi))
            # b) substitute
            expr = expr.subs(derivative, derivative.as_finite_difference(dxi, wrt=xi)).expand()

    return expr


def finite_difference_system(system_pde):
        """ Compute the finite difference discretization of a system of pdf which represent evolution equation

         Comment
         -------

            Only convert rhs of each equations, the lhs beeing the trend.

         """

        fd_system = []

        for pde in system_pde:
            # -1- extract lhs/rhs
            lhs, rhs = pde.args
            # -2- finite difference of rhs
            rhs = finite_difference(rhs)
            # -3- update the system
            fd_system.append(Eq(lhs, rhs))

        return fd_system


def get_displacement(function, dx=None):
    """ Return infinitesimal displacement in dx base

    Example
    -------

        >>> t, x, y = symbols('t x y')
        >>> dt, dx, dy = symbols('dt dx dy')
        >>> U = Function('U')(t,x,y)
        >>> U.subs({x:x+dx, y:y-3dy/2})
        U(t,x+dx,y-3dy/2)
        >>> get_displacement(U)
        (0,1,-3/2)
    """
    if dx is None:
        x, dx = get_function_coordinates(function)

    return tuple(arg.coeff(dxi) for arg, dxi in zip(function.args, dx))


def regularize_finite_difference(finite):
    """ Transform finite difference written en staggered grid into regular grid

    Example
    -------

        (U(dx/2+x)-U(-dx/2+x))/dx   is replaced by (U(dx+x) - U(dx-x))/(2dx)

    """

    from sympy import Rational, fraction, Integer, Function

    warnings.warn("regularize_finite_difference should be removed in future", DeprecationWarning)

    for function in finite.atoms(Function):

        # -1- Find coordinate system `x` adapted to the function
        x, dx = get_function_coordinates(function)

        # -2- Find decomposition of the infinitesimal displacement in the `dx` base
        displacement = get_displacement(function, dx)

        # -3- Compute new displacement and scaling
        new_displacement = []
        scaling = Integer(1)
        for coeff in displacement:
            if isinstance(coeff, Rational):
                numerator, denominator = fraction(coeff)
                new_displacement.append(numerator)
                scaling *= Rational(1, denominator)
            elif isinstance(coeff, Integer):
                new_displacement.append(coeff)
            else:
                raise ValueError(f"{coeff} is not Integer or Rational")

        # -4- Replace old function by the new one
        new_args = [xi + coeff * dxi for xi, coeff, dxi in zip(x, new_displacement, dx)]
        finite = finite.subs({function: scaling * function.func(*new_args)})

    return finite


def regularize_finite_difference_system(system):
    """ Transform a finite difference system into a regularized system """
    warnings.warn("regularize_finite_difference_system should be removed in future", DeprecationWarning)
    regular = []
    for eq in system:
        lhs, rhs = eq.args
        rhs = regularize_finite_difference(rhs)
        regular.append(Eq(lhs, rhs))
    return regular


def slice_coding_rule(k, step):
    return f'self.index({k},{step})'


def code_regular_function(function, slice_coding_rule):
    """

    :param function:
    :param slice_coding_rule:
    :return:


    """
    from sympy import Integer
    from .tool import clean_latex_name

    # -1- get information about the function
    x, dx = get_function_coordinates(function)
    displacement = get_displacement(function)

    # -2- code function name
    code = clean_latex_name(function.func)

    # -3- Add tag for non zero displacement

    # a) Tag for time step
    if 't' in [str(xi) for xi in x]:
        assert str(x[0]) == 't'

        step = displacement[0]
        assert isinstance(step, Integer), f"displacement is not on regular grid for function: {function}"

        if step != 0:
            tag = f'm{abs(step)}' if step < 0 else f'p{abs(step)}' if step > 0 else ''
            code += tag

        x = x[1:]
        displacement = displacement[1:]

    # b) Tag for spatial step
    if any([step != 0 for step in displacement]):

        # Prefix for opening array
        code += '[np.ix_('
        for k, step in enumerate(displacement):
            assert isinstance(step, Integer), f"displacement is not on regular grid for function: {function}"

            code += slice_coding_rule(f"'{x[k]}'", step) + ','

        # Suffix for closing array
        code = code[:-1]
        code += ')]'

    return code


def dx_coding_rule(coord):
    return f"self.dx['{coord}']"


def code_finite_difference(finite, slice_coding_rule, dx_coding_rule):
    """ Transform a finite difference sympy expression into a python code.

    :param finite:
    :param slice_coding_rule:
    :return:
    """
    from sympy import Function


    code = str(finite)


    loc_x_dx = []

    # -1- Replace functions
    for function in finite.atoms(Function):
        x, dx = get_function_coordinates(function)
        for xi,dxi in zip(x,dx):
            if (xi,dxi) not in loc_x_dx:
                loc_x_dx.append((xi,dxi))
        code = code.replace(str(function), code_regular_function(function, slice_coding_rule))

    # -2- Replace dx's
    for x_dx in loc_x_dx:
        xi, dxi = x_dx
        code = code.replace(str(dxi), dx_coding_rule(xi))

    return code


class Code(object):

    def __init__(self,expr):
        self.expr = expr
        self._code = None

    def __repr__(self):
        return self.code

    @staticmethod
    def slice_coding_rule(k,step):
        return f'self.index({k},{step})'

    @staticmethod
    def dx_coding_rule(coord):
        return f"self.dx['{coord}']"

    def _render_function(self, function):
        """
        Transform a function into code

        Example
        -------

            >>> f = Function('f^u_xy')(x,y)
            >>> Code(f).code
            f_u_xy
            >>> f = Function('f^u_xy')(t,x,y) # Eliminate time
            >>> Code(f).code
            f_u_xy
            >>> f = Function('f^u_xy')(t+dt,x,y) # Add tag for +dt time
            >>> Code(f).code
            fp1_u_xy
            >>> f = Function('f^u_xy')(t-2*dt,x,y) # Add tag for +dt time
            >>> Code(f).code
            fm2_u_xy
            >>> f = Function('f^u_xy')(t,x+dx,y)
            >>> Code(f).code
            f_u_xy[np.ix_(self.index('x',1),self.index('y',0))]

        :param function:
        :return:
        """
        from sympy import Integer
        from .tool import clean_latex_name

        # -1- get information about the function
        x, dx = get_function_coordinates(function)
        displacement = get_displacement(function)

        # -2- code function name
        code = clean_latex_name(function.func)

        # -3- Add tag for non zero displacement

        # a) Tag for time step
        if 't' in [str(xi) for xi in x]:
            assert str(x[0]) == 't'

            step = displacement[0]
            assert isinstance(step, Integer), f"displacement is not on regular grid for function: {function}"

            if step != 0:
                tag = f'm{abs(step)}' if step < 0 else f'p{abs(step)}' if step > 0 else ''
                code += tag

            x = x[1:]
            displacement = displacement[1:]

        # b) Tag for spatial step
        if any([step != 0 for step in displacement]):

            # Prefix for opening array
            code += '[np.ix_('
            for k, step in enumerate(displacement):
                assert isinstance(step, Integer), f"displacement is not on regular grid for function: {function}"

                code += self.slice_coding_rule(f"'{x[k]}'", step) + ','

            # Suffix for closing array
            code = code[:-1]
            code += ')]'

        return code

    @property
    def code(self):
        if self._code is None:
            self._code = code_finite_difference(self.expr,self.slice_coding_rule,self.dx_coding_rule)
        return self._code
