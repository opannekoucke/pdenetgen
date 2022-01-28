from sympy import Eq
from .util import PDESystem, get_trend, get_derivative_partial_orders, FunctionSymbol, ScalarSymbol

from jinja2 import Template

from .tool import clean_latex_name

class MetaFunction(object):

    def __init__(self, function, index=None):
        self._function = function
        self.as_symbolic = function
        self.as_keyvar = self._set_as_keyvar()
        self.as_str = self._set_as_str()
        self.index = index

    def _set_as_keyvar(self):
        return clean_latex_name(self.as_symbolic.func)

    def _set_as_str(self):
        return str(self.as_symbolic)

class MetaPrognosticFunction(MetaFunction):

    def __init__(self, function, index=None):
        super().__init__(function, index)
        self.trend = self._set_trend()

    def _set_trend(self):
        return 'd'+self.as_keyvar


class MetaDerivative(object):

    def __init__(self, derivative):
        self._partial_orders = get_derivative_partial_orders(derivative)
        self.as_symbolic = derivative
        self.function = MetaFunction(self.as_symbolic.args[0])
        self.as_keyvar = self._set_as_keyvar()
        self.as_str = self._set_as_str()
        self.as_partial_orders = self._set_as_partial_orders()
        self.order = sum(partial_order for partial_order in self._partial_orders.values())
        self.as_code = self._set_as_code()

    def _set_as_keyvar(self):
        keyvar = "_".join(f'{coordinate}_o{partial_order}'
                          for coordinate, partial_order in self._partial_orders.items())
        keyvar = 'D' + self.function.as_keyvar + '_' + keyvar
        return keyvar

    def _set_as_partial_orders(self):
        return ",".join(f"'{coordinate}',{partial_order}"
                        for coordinate, partial_order in self._partial_orders.items())

    def _set_as_str(self):
        return str(self.as_symbolic)

    def _set_as_code(self):
        raise NotImplementedError

class MetaConstant(object):
    def __init__(self, constant):
        self.as_symbolic = constant
        self.as_keyvar = self._set_as_keyvar()
        self.as_str = self._set_as_str()

    def _set_as_keyvar(self):
        return clean_latex_name(self.as_symbolic)

    def _set_as_str(self):
        return str(self.as_symbolic)

class MetaCoordinate(object):

    def __init__(self, coordinate, index=0):
        self.as_symbolic = coordinate
        self.as_str = self._set_str()
        self.as_keyvar = self._set_as_keyvar()
        self.index = index

    def _set_as_keyvar(self):
        return clean_latex_name(self.as_symbolic)

    def _set_str(self):
        return str(self.as_symbolic)


class MetaFactory(object):

    def __new__(cls):
        pass

class ModelBuilder(object):
    """ Model Builder from symbolic system of partial differential equations

    This builder is limited to simple functions do not consider mix of 2D/3D, but can handle 2D functions as well as 3D functions

    .. Description :

        Given a system of PDE, the translation to code is made as follows:

        1. Extract all spatial derivatives
        2. Create a code representation of equations, where spatial derivative are replaced by an intermediate variable
            -> the pseudo_algebraic_system
        3. Create the code from the 'pseudo_algebraic_system'

        Code is then generated from jinja2 template. The core of the code is the definition of the 'trend' method.

        In 'trend' the computation of the spatial derivative is first made, stored in intermediate variables tagged
        by using the convention defined in the 'class _Derivative'.

    """
    template_licence = '#Bug Licence.. \n' #None
    template_header = None
    template_footer = None
    _default_name = 'NumModel'
    template_body = None
    meta_factory = None

    def __init__(self, pde_system, class_name=None):

        if not isinstance(pde_system, PDESystem):
            pde_system = PDESystem(pde_system)

        self.pde_system = pde_system

        if class_name is None:
            if pde_system.name == '':
                class_name = self._default_name
            else:
                class_name = pde_system.name

        self.class_name = class_name
        self._time_symbol = self.pde_system._time_symbol

        self._functions = None
        self._prognostic_functions = None
        self._constant_functions = None
        self._exogenous_functions = None
        self._constants = None
        self._spatial_derivatives = None
        self._spatial_coordinates = None
        self._pseudo_algebraic_system = None
        self._trend_code = None
        self._code = None

    @property
    def functions(self):
        if self._functions is None:
            self._functions = [MetaFunction(function, index)
                               for index, function in enumerate(self.pde_system.functions)]
        return self._functions

    @property
    def _subs_trend(self):
        """ Associate a progonstic function to its coding rule """


    @property
    def pseudo_algebraic_system(self):
        if self._pseudo_algebraic_system is None:
            self._pseudo_algebraic_system = self._make_pseudo_algebaraic_system()
        return self._pseudo_algebraic_system

    @property
    def trend_code(self):
        if self._trend_code is None:
            self._trend_code = self._make_trend_code()
        return self._trend_code

    def _create_meta_function(self, function, index):
        return MetaFunction(function, index)
    def _create_meta_prognostic_function(self, function, index):
        return MetaPrognosticFunction(function, index)
    def _create_meta_derivative(self, derivative):
        return MetaDerivative(derivative)
    def _create_meta_coordinate(self, coordinate, index):
        return MetaCoordinate(coordinate, index)
    def _create_meta_constant(self, constant):
        return MetaConstant(constant)


    @property
    def prognostic_functions(self):
        if self._prognostic_functions is None:

            self._prognostic_functions = [self._create_meta_prognostic_function(function, index)
                                          for index, function in enumerate(self.pde_system.prognostic_functions)]

        return self._prognostic_functions

    @property
    def constant_functions(self):
        if self._constant_functions is None:
            self._constant_functions = [self._create_meta_function(function, index)
                                        for index, function in enumerate(self.pde_system.constant_functions)]
        return self._constant_functions

    @property
    def exogenous_functions(self):
        if self._exogenous_functions is None:
            self._exogenous_functions = [self._create_meta_function(function, index)
                                        for index, function in enumerate(self.pde_system.exogenous_functions)]
        return self._exogenous_functions

    @property
    def constants(self):
        if self._constants is None:
            self._constants = [self._create_meta_constant(constant) for constant in self.pde_system.constants]
        return self._constants

    @property
    def spatial_derivatives(self):
        if self._spatial_derivatives is None:
            self._spatial_derivatives = [self._create_meta_derivative(derivative)
                                         for derivative in self.pde_system.spatial_derivatives]
        return self._spatial_derivatives

    @property
    def spatial_coordinates(self):
        if self._spatial_coordinates is None:
            self._spatial_coordinates = [self._create_meta_coordinate(coordinate, index)
                                         for index, coordinate in enumerate(self.pde_system.spatial_coordinates)]
        return self._spatial_coordinates

    def _make_pseudo_algebaraic_system(self):
        """ Create the pseudo algebraic system associted with the PDE system
            this represent the equation where rhs are algebraic + standard functions (sin/cos/tan/ .. )
        """

        orders = list(range(1, self.pde_system.derivative_max_order + 1))
        orders.reverse()

        pseudo_algebraic_system = []

        for equation in self.pde_system.equations: # todo : should be prognostic equation .. in place of equations
            # -1- Convert rhs into string
            lhs, rhs = equation.args

            if lhs != get_trend(equation):
                raise ValueError(f"Equation {lhs}= .. is not prognostic.")

            # -2- Replace all Derivative in the rhs string from the higher to the lower order derivative
            for order in orders:
                for derivative in self.spatial_derivatives:
                    # a) find all derivative of order `order`.
                    if derivative.order != order:
                        continue
                    # b) substitute all these derivative in the rhs.
                    rhs = rhs.subs(derivative.as_symbolic, FunctionSymbol(derivative.as_keyvar))

            # -3- Replace all functions
            for function in self.functions:
                rhs = rhs.subs(function.as_symbolic, FunctionSymbol(function.as_keyvar))

            # -4- Replace all constants
            for constant in self.constants:
                rhs = rhs.subs(constant.as_symbolic, ScalarSymbol(constant.as_keyvar))

            pseudo_algebraic_system.append(Eq(lhs,rhs))

        return pseudo_algebraic_system

    def _make_trend_code(self):
        """ Create the code for the trend from the pseudo algebraic system """
        subs_trend = {function.as_symbolic:function.trend
                      for function in self.prognostic_functions}
        trend_code = []
        for equation in self.pseudo_algebraic_system:
            # 1. extract lhs, rhs
            lhs, rhs = equation.args
            # 2. subs lhs using coding rule
            function = lhs.args[0]
            lhs = subs_trend[function]
            # 3. implement the code line: set the lhs **np.array** from the rhs
            code_line = str(lhs)+'[:] = '+str(rhs)
            # 4. Add to code.
            trend_code.append(code_line)

        return trend_code

    @property
    def code(self):
        if self._code is None:
            self._render()
        return self._code

    @property
    def templates(self):
        return [self.template_header, self.template_body, self.template_footer]

    def _render(self):
        """ Render the template to produce the python source code """

        #
        # Faire une boucle sur les templates pour alimenter le code
        #

        code = ''''''

        for template in self.templates:
            template = Template(template)

            # -1- Render code with temporay tags `self.tmp_tag`
            code += template.render(
                class_name=self.class_name,
                prognostic_functions=self.prognostic_functions,
                exogenous_functions=self.exogenous_functions,
                exogenous_functions_flag=self.exogenous_functions != [],
                coordinates=self.spatial_coordinates,
                constants=self.constants,
                constants_flag=self.constants != [],
                constant_functions=self.constant_functions,
                constant_functions_flag=self.constant_functions != [],
                spatial_derivatives=self.spatial_derivatives,
                trend_code=self.trend_code
            )

        self._code = code

    def write_module(self):
        # -1- Render the code
        code = self.template_licence + self.code

        # -2- Save code in a module
        python_module = self.class_name.lower()
        python_file = python_module + '.py'
        with open(python_file, 'w') as file:
            for line in code:
                file.write(line)

        self.module_name = python_module
        self.module_file = python_file
        print(f"class {self.class_name} has been written in module {python_module} in file {python_file}")


