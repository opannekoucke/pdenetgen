"""
    Symbolic tools add for facilitating symbolic computation with sympy.
"""

from sympy import Derivative, Function, Symbol, symbols, Add
from .tool import clean_latex_name
import sympy

from .constants import t as time_coordinate

omega = Symbol('omega')


class ScalarSymbol(Symbol):
    """
    Model a scalar value in a dynamics
    """
    pass

class TrainableScalar(Symbol):
    """
    Trainable Scalar for data-driven and physic-informed dynamics

    Description
    -----------

    init_value  :   Set the initial value of the unknown quantity.
                    When init_value = None (default value) then, the initial value
                    is a sample of the Gaussian law of mean 'mean' and standard deviation 'stddev'
    mean        :   Mean of the Gaussian sample to set the initial value (when init_value=None)
    stddev      :   Standard deviation of the Gaussian sample (when init_value=None)
    wl2         :   l2 penalty used when not wl2 is not None.

    Example
    -------

    >>> a = TrainableScalar('a') # Configure the unknown quantity as an unknown initialized by a Gaussian (0,1)
    >>> b = TrainableScalar('b', init_value= 0.001)
    """

    options = ['init_value','use_bias','mean','stddev','seed','wl2']

    def __new__(cls, name, init_value=None, use_bias=False, mean=0., stddev=1., seed=None, wl2=None, **assumptions):
        instance = super(TrainableScalar, cls).__new__(cls, name, **assumptions)
        instance.init_value = init_value
        instance.use_bias = use_bias
        instance.mean = mean
        instance.stddev = stddev
        instance.seed = seed
        instance.wl2 = wl2
        return instance


class FunctionSymbol(Symbol): pass


def upper_triangle(matrix):
    """ Returns the upper triangle of a matrix """
    upper = []
    for i in range(matrix.shape[0]):
        for j in range(i, matrix.shape[1]):
            upper.append(matrix[i, j])
    return upper

class Matrix(sympy.Matrix):

    @property
    def upper_triangle(self):
        upper = []
        for i in range(self.shape[0]):
            for j in range(i, self.shape[1]):
                upper.append(self[i, j])
        return upper


class Eq(sympy.Eq):
    """
    Extension of sympy.Eq which facilitates handling of equations.

    Warning
    -------

        Multiplication of Equation should be made by the right:

        >>> eq = Eq(a,b)
        >>> eq * Derivative(u,t)
        a*Derivative(u,t) = b*Derivative(u,t)

    Exemple
    -------

        >>> from sympy import symbols
        >>> a, b, c, d = symbols('a b c d')
        >>> eq1 = Eq(a,b)
        >>> eq2 = Eq(c,d)
        >>> -eq1
        -a = -b
        >>> eq1 + eq2
        a+c = b+d
        >>> eq1 - eq2
        a-c = b-d
        >>> Integer(2)*eq1
        2*a = 2*b
        >>> eq3 = eq1 + eq2
        >>> eq3 = eq3.isolate(a)
        a = b+d-c
        >>> eq3.subs(b,0)
        a = d-c
        >>> eq3*a*c  # Needs right multiplication !!!
        a**2*c = a*c*(b-c+d)
        >>> eq3 *= a*c
        >>> display(eq3)
        a**2*c = a*c*(b-c+d)

    """

    def __mul__(self, alpha):
        return self.__class__(*[alpha * self.args[k] for k in range(2)])

    def __imul__(self, alpha):
        if isinstance(alpha, Eq):
            return self.__class__(*[self.args[k] * alpha.args[k] for k in range(2)])
        else:
            return self.__class__(*[alpha * self.args[k] for k in range(2)])

    def __add__(self, eq):
        return self.__class__(*[self.args[k] + eq.args[k] for k in range(2)])

    def __neg__(self):
        return self.__class__(*[-self.args[k] for k in range(2)])

    def __sub__(self, eq):
        return self.__class__(*[self.args[k] - eq.args[k] for k in range(2)])

    def subs(self, *args, **hints):
        eq = super().subs(*args, **hints)
        return self.__class__(*eq.args)

    def doit(self, *args, **hints):
        eq = super().doit(*args, **hints)
        return self.__class__(*eq.args)

    def expand(self, *args, **hints):
        eq = super().expand(*args, **hints)
        return self.__class__(*eq.args)

    def isolate(self, expr):
        """ Isolate an expression from an equation

        Example
        -------

            >>> eq = Eq(Derivative(u,t)+u*Derivative(u,x),0)
            >>> eq.isolate(Derivative(u,t))
            Derivative(u,t) = -u*Derivative(u,x)
        """
        from sympy import solve
        return self.__class__(expr, solve(self, expr)[0])


class PDESystem(object):
    """ Symbolic system of partial differential equations
    """

    _time_symbol = time_coordinate

    def __init__(self, equations, name=''):

        if not isinstance(equations, list):
            equations = [equations]

        self.equations = equations

        self.name = name

        # Set endogenous/exogenous fields
        self.functions = self._get_functions()
        self.prognostic_functions = self._get_prognostic_functions()
        self.constant_functions = self._get_constant_functions()
        self.exogenous_functions = self._get_exogenous_functions()

        # Set spatio/temporal coordinates
        self.coordinates = self._get_coordinates()
        self.spatial_coordinates = self._get_spatial_coordinates()
        self.time_coordinate = self.coordinates[0]

        # Set trainable scalar
        self.trainable_scalars = self._get_trainable_scalars()

        # Set constants
        self.constants = self._get_constants()

        # Set derivatives which appears in the system of pde
        self.spatial_derivatives = self._get_spatial_derivatives()


        self.derivative_max_order = max([0]+[derivative.derivative_count for derivative in self.spatial_derivatives])

    def __iter__(self):
        """ Build an iterator on the system equations """

        def iterator():
            for equation in self.equations:
                yield equation

        return iterator()

    def __getitem__(self, item):
        """ Get k'th equation of the system """
        return self.equations[item]

    def __repr__(self):
        """ Sumup of the PDESystem """
        """
        todo : add trainable_scalar
        """
        name = f"'{self.name}'" if self.name != '' else ''
        prognostic_functions = ", ".join([str(function)
                                          for function in self.prognostic_functions])
        exogenous_functions = ", ".join([str(function)
                                         for function in self.exogenous_functions])
        constant_functions = ", ".join([str(function)
                                        for function in self.constant_functions])
        constants = ", ".join([str(clean_latex_name(constant)) for constant in self.constants])
        trainable_scalars = ",".join([str(clean_latex_name(constant)) for constant in self.trainable_scalars])

        string = f'''PDE System {name}:
        prognostic functions : {prognostic_functions}
        constant functions   : {constant_functions}
        exogeneous functions : {exogenous_functions}
        constants            : {constants}
        '''
        if self.trainable_scalars != set():
            string += f'''
        trainable scalars    : {trainable_scalars}
            '''

        return string

    def _get_functions(self):
        """ Extract all functions present in the system of pde """

        functions = set()

        for equation in self.equations:
            functions.update(equation.atoms(Function))

        return functions

    def _get_prognostic_functions(self):
        """ Extract prognostic functions from the system of pde """

        prognostic_functions = []

        for equation in self.equations:
            derivatives = equation.atoms(Derivative)
            for derivative in derivatives:
                coordinates = derivative.args[1].atoms(Symbol)
                if self._time_symbol in coordinates:
                    prognostic_functions.append(derivative.args[0])

        return prognostic_functions

    def _get_constant_functions(self):
        # Selects constant variables (which is a function of time free coordinate)

        constant_functions = set()

        for variable in self.functions.difference(self.prognostic_functions):
            coordinates = variable.args[0].atoms(Symbol)
            if self._time_symbol not in coordinates:
                constant_functions.update({variable})

        return constant_functions

    def _get_exogenous_functions(self):
        """ Extract diagnostic / exogenous functions

        .. Warning:
            there is not difference here between diagnostic and exogenous functions.
        """

        # Extract diagnostic/exogenous functions from all functions
        exogenous_functions = self.functions.difference(self.prognostic_functions)
        exogenous_functions.difference_update(self.constant_functions)

        return exogenous_functions

    def _get_coordinates(self):
        """ Extract spatial coordinates from arguments of prognostic functions """
        coordinates = ()
        for function in self.prognostic_functions:
            args = function.args
            if len(args) > len(coordinates):
                coordinates = args
        return coordinates

    def _get_spatial_coordinates(self):
        assert self.coordinates[0] == self._time_symbol
        return self.coordinates[1:]

    def _get_constants(self):
        ''' List all Constant from a system of PDE '''

        constants = set()  # use set to only select one single sample of a given constant (no duplication)
        for equation in self.equations:
            rhs = equation.args[1]
            trainable = rhs.atoms(TrainableScalar)
            # Only retain constants that are not coordinate nor trainable
            constants.update(rhs.atoms(Symbol).difference(self.coordinates).difference(trainable))

        return constants

    def _get_trainable_scalars(self):
        ''' List all Constant from a system of PDE '''

        trainable_scalars = set()  # use set to only select one single sample of a given constant (no duplication)
        for equation in self.equations:
            rhs = equation.args[1]
            trainable_scalars.update(rhs.atoms(TrainableScalar))

        return trainable_scalars

    def _get_spatial_derivatives(self):
        ''' List all Derivative from a system of PDE '''

        # -1- Check all derivatives
        spatial_derivatives = set()  # use set to only select one single sample of a given derivative (no duplication)
        # loop over equations
        for equation in self.equations:
            # extract constant from rhs of equations
            rhs = equation.args[1]
            spatial_derivatives.update(rhs.atoms(Derivative))

        return spatial_derivatives

    @property
    def finite_difference(self):
        """ Compute the finite difference discretization of a system of pde """
        from .finite_difference import finite_difference, Eq

        orders = list(range(1, self.derivative_max_order + 1))
        orders.reverse()

        finite_difference_system = []

        for equation in self.equations:
            # -1- extract lhs/rhs
            lhs, rhs = equation.args
            # -2- finite difference of rhs
            rhs = finite_difference(rhs)
            # -3- update the system
            finite_difference_system.append(Eq(lhs, rhs))

        return finite_difference_system


def remove_eval_derivative(expr):
    """
    Remove substituted derivative in an expression.
    :param expr:
    :return:
    """
    eval_terms = expr.atoms(sympy.Subs)
    eval_derivative = {}
    for eval_term in eval_terms:
        eval_derivative[eval_term] = eval_term.args[0].subs({
            key: value for key, value in zip(eval_term.args[1], eval_term.args[2])
        })
    return expr.subs(eval_derivative).doit()


def get_function_coordinates(function):
    """ Given a function evaluated from finite difference: return the coordinate system and its differential forms

    Example
    -------

        >>> t, x, y = symbols('t x y')
        >>> dt, dx, dy = symbols('dt dx dy')
        >>> U = Function('U')(t,x,y)
        >>> U.subs({x:x+dx, y:y-3dy/2})
        U(t,x+dx,y-3dy/2)
        >>> get_function_coordinates(U)
        ([t,x,y],[dt,dx,dy])

    """
    from sympy import Symbol, symbols

    if not isinstance(function, Function):
        raise ValueError("'function' should be a function")

    x = []
    dx = []

    for arg in function.args:

        coordinate = list(arg.atoms(Symbol))

        if len(coordinate) == 2:
            xi, dxi = coordinate
            if str(xi)[0] == 'd':
                xi, dxi = dxi, xi
        elif len(coordinate) == 1:
            xi = coordinate.pop()
            dxi = symbols('d' + str(xi))
        else:
            raise ValueError(f"Too many symbols {arg} for function {function}")

        x.append(xi)
        dx.append(dxi)

    return x, dx


def get_coordinate_system(expr):
    """ Return the coordinates system used as a tuple ([xi],[dxi])

    For random variable, 'omega' is not considered.

    Example
    -------

    >>> t,x,y = symbols('t x y')
    >>> u = Function('u')(t,x,y)
    >>> get_coordinates(u)
    ([t, x, y],[dt, dx, dy])

    For a random variable: omega is ignored

    >>> u = Function('u')(t,x,y, omega)
    >>> get_coordinates(u)
    ([t, x, y],[dt, dx, dy])

    """

    # todo: update to apply for equation defined by 'Eq'
    # 1. Get the coordinate system used.
    functions = expr.atoms(Function)

    # 2. Extract the coordinate system of each function.
    coordinates = [[], []]
    for function in functions:
        coordinate = get_function_coordinates(function)
        for xi, dxi in zip(coordinate[0], coordinate[1]):
            if xi is omega:
                continue
            if not xi in coordinates[0]:
                coordinates[0].append(xi)
                coordinates[1].append(dxi)

    return coordinates

def get_trend(equation):
    """
    Return all temporal trend from an equation ie Derivative(field,t)
    :param equation:
    :return:
    """
    trends = set()
    for derivative in equation.atoms(Derivative):
        if derivative.args[1] == (time_coordinate, 1):
            trends.add(derivative)
    if len(trends) == 1:
        trends = trends.pop()
    return trends

def get_derivative_partial_orders(derivative):
    """ Return the dictionary of partial orders

    Example
    -------

        >>> expr = Derivative(U,x,2,y,1)
        >>> expr.args[1:]
        ((x,2),(y,1))
        >>> derivative_order_tag(expr.args[1:])
        ('x,2,y,1','x_o2_y_o1')

    """

    all_orders = derivative.args[1:]

    partial_orders = {}

    if isinstance(all_orders[0], Symbol):
        # e.g. case orders=(x,1)
        coordinate, partial_order = all_orders
        partial_orders[coordinate] = partial_order

    else:
        # e.g. case orders=((x,1),(y,2))
        for sub_order in all_orders:
            coordinate, partial_order = sub_order
            partial_orders[coordinate] = partial_order

    return partial_orders

def get_total_order(derivative):
    """ Compute the total order of a derivative

    Example:
        >>> from sympy import Derivative, Function, symbols
        >>> x,y = symbols('x y')
        >>> get_total_order(Derivative( Function('u')(x,y), x,2,y,4))
        6
    """
    partial_orders = get_derivative_partial_orders(derivative)
    return sum([value for key,value in partial_orders.items()])

class CoordinateSystem(object):
    """ Facilitate handling of coordinate system """

    def __init__(self, coords):
        self._coords = coords

    def __iter__(self):
        """ Make iterator on coordinates """

        def iterator():
            for xi in self._coords:
                yield xi

        return iterator()

    def gradient(self, field):
        # 1) verify that field in function of coordinate system
        self._check_compatible(field)
        # 2) Compute the gradient
        return Matrix([Derivative(field, xi) for xi in self])

    def div(self, vector):
        # 1) Verify that vector field is compatible
        for component in vector:
            self._check_compatible(component)
        # 2) Compute the gradient
        return sum([Derivative(vi, xi) for vi, xi in zip(vector, self)])

    def is_compatible(self, scalar_function):
        """ Verify if coordinates of scalar_function includes the coordinate system """
        coords = set(self._coords)
        function_coords = set(scalar_function.args)
        if coords.issubset(function_coords):
            return True
        else:
            return False

    def _check_compatible(self, expr):
        for function in expr.atoms(Function):
            if not self.is_compatible(function):
                raise ValueError(f'field args {function.args} not compatible with coordinates {self._coords}')


class CompareEquation(object):
    """ Compare two symbolic equations, with a comparison of their lhs/rhs """

    def __init__(self, eq1, eq2):
        self.equations = [eq1, eq2]
        self.lhs = CompareExpression(*[eq.args[0] for eq in self.equations])
        self.rhs = CompareExpression(*[eq.args[1] for eq in self.equations])


class CompareExpression(object):
    """ Compare two symbolic expression to understand:
        - what they have in common,
        - what are their differences
    """

    def __init__(self, exp1, exp2):
        self.expressions = [exp1.expand(), exp2.expand()]
        self._sets = None

    @property
    def sets(self):
        """ Return the set of sub terms from expansion """
        if self._sets is None:
            sets = []
            for exp in self.expressions:
                if isinstance(exp, Add):
                    sets.append(set(exp.args))
                else:
                    sets.append(set([exp]))

            self._sets = sets

        return self._sets

    @property
    def common(self):
        """ Return terms in common """
        s1, s2 = self.sets
        return s1.intersection(s2)

    @property
    def specific(self):
        """ Return terms wich are specific to each expressions """
        return [s.difference(self.common) for s in self.sets]
