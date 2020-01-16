from .util import Eq, get_coordinates


def finite_difference(expr, regular_grid=True):
    """ Finite differenciate derivative in an expression

    Documentation
    -------------

        from sympy, see: Fornberg1988MC

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

    if regular_grid:
        expr = regularize_finite_difference(expr)

    return expr


def finite_difference_system(system_pde, regular_grid=True):
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

        if regular_grid:
            fd_system = regularize_finite_difference_system(fd_system)

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
        x, dx = get_coordinates(function)

    return tuple(arg.coeff(dxi) for arg, dxi in zip(function.args, dx))


def regularize_finite_difference(finite):
    """ Transform finite difference written en staggered grid into regular grid

    Example
    -------

        (U(dx/2+x)-U(-dx/2+x))/dx   is replaced by (U(dx+x) - U(dx-x))/(2dx)

    """

    from sympy import Rational, fraction, Integer, Function

    for function in finite.atoms(Function):

        # -1- Find coordinate system `x` adapted to the function
        x, dx = get_coordinates(function)

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
    x, dx = get_coordinates(function)
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
        x, dx = get_coordinates(function)
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
        x, dx = get_coordinates(function)
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
