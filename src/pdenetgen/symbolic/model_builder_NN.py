from sympy import Derivative, Symbol, Function
import numpy as np
from .nn_builder import CodeSkeleton, KerasCodingRule, MetaLayer
from .model_builder import ModelBuilder
from .model_builder import MetaDerivative as SuperMetaDerivative
from .model_builder import MetaFunction as SuperMetaFunction
from .model_builder import MetaPrognosticFunction as SuperMetaPrognosticFunction
from .model_builder import MetaCoordinate as SuperMetaCoordinate
from .util import get_function_coordinates
from .finite_difference import finite_difference, get_displacement
from .constants import t as time_coordinate

class MetaFunction(SuperMetaFunction):

    def __init__(self, function, index=None):
        super().__init__(function, index)
        self.input_shape = self._set_input_shape()

    def _set_input_shape(self):
        """ Return the tuple of the NN shape that should be associated to a given function depending on its coordinates

        .. Example:
            if coordinate system is (x,y,z) then input_shape should be as follows
                function        input_shape (is a tuple)    # additionnal 1 is for use in ConvNet
                f(t,x,y,z)      (shape_x, shape_y, shape_z) # additionnal 1 is for use in ConvNet
                f(x,y)          (shape_x, shape_y)          # additionnal 1 is for use in ConvNet
                f(t,x)          (shape_x,)
                f(t)            (1,)
        """
        #t = self._time_symbol
        #t = Symbol('t')
        t = time_coordinate
        coordinates = self.as_symbolic.args

        if coordinates == (t,):
            return (1,)

        shape = []
        for coordinate in coordinates:
            if coordinate is t:
                continue
            shape.append(MetaCoordinate(coordinate).input_shape)

        return tuple(shape)+(1,)


class MetaPrognosticFunction(MetaFunction,SuperMetaPrognosticFunction):

    def _set_trend(self):
        return 'trend_'+self.as_keyvar


class Kernel(object):

    _time_symbol = Symbol('t')

    def __init__(self,derivative):
        self._derivative = derivative
        self._function = derivative.args[0]
        self._coordinates,self._dxs = get_function_coordinates(self._function)
        self._suppress_time = False
        self._dimension = len(self._coordinates)
        if self._time_symbol == self._coordinates[0]:
            self._suppress_time = True
            self._dimension -= 1
            self._coordinates = self._coordinates[1:]
            self._dxs = self._dxs[1:]
        self._stencil = self._compute_stencil()
        max_index = max(np.abs(index).max() for index in self._stencil)
        self._size = self._dimension * (2 * max_index + 1,)
        self._center = np.array(self._dimension * (max_index,))
        self._kernel = self._set_kernel()
        self.as_code = self._set_as_code()

    @property
    def dimension(self):
        return self._dimension
    @property
    def size(self):
        return self._size

    def _set_as_code(self):
        kernel = self._reformat_kernel_str(self._kernel)
        return "np.asarray("+ kernel +f").reshape({self._size}+(1,1))"

    def _set_kernel(self):
        # 1. Create the array of string using the dimension of the kernel
        kernel = np.zeros(self._size, dtype='U100')
        kernel[:] = '0.0'

        # 2. Set the kernel from te stencil
        for index,weight in self._stencil.items():
            # a) Convert the weight in code
            weight = str(weight)
            weight = self._replace_dx(weight)
            # b) Recenter the index
            index = tuple(np.array(index)+self._center)
            # c) Set the kernel
            kernel[index] = weight
        return kernel

    def _compute_stencil(self):
        """ Compute the stencil associated with the finite difference discretization of a derivative

        .. example:
            >>> t,x,y = symbols('t x y')
            >>> u = Function('u')(t,x,y)
            >>> compute_stencil(Derivative(u,x))
            {(1, 0): 1/(2*dx), (-1, 0): -1/(2*dx)}

        """
        # 1. Compute the finite difference **in regular grid**
        finite = finite_difference(self._derivative)

        # 2. Extract function
        functions = list(finite.atoms(Function))

        # 3. Extract displacement vector which corresponds to the kernel_index
        stencil_index = [get_displacement(function) for function in functions]
        # 4. Suppress first index if time
        if self._suppress_time:
            # Eliminate first elment
            stencil_index = [index[1:] for index in stencil_index]
        # 5. Extract the weight
        stencil_weight = [finite.coeff(function) for function in functions]
        # 6. Construct the kernel and return
        stencil = {index:weight for index, weight in zip(stencil_index,stencil_weight)}
        return stencil

    @staticmethod
    def slice_coding_rule(coord, step: int):
        return f'self.index({coord},{step})'

    @staticmethod
    def dx_coding_rule(coord):
        return f"self.dx[self.coordinates.index('{coord}')]"

    def _replace_dx(self, string):
        for xi, dxi in zip(self._coordinates, self._dxs):
            string = string.replace(str(dxi),self.dx_coding_rule(xi))
        return string

    @staticmethod
    def _reformat_kernel_str(kernel):
        # 1. Convert to string.
        m = str(kernel)

        # 2. Apply replace rules.
        m = m.replace("   ","")
        m = m.replace("\n\n","\n")
        m = m.replace("\n",",\n")
        m = m.replace('"','')
        m = m.replace(' ',",")
        m = m.replace(',,','  ')
        m = m.replace(',[',' [')
        m = m.replace("'0.0'","0.0")
        m = m.replace('\n,',"\n")

        return m

class MetaDerivative(SuperMetaDerivative):

    def _set_as_code(self):


        code = []

        # 1. Construct the code lines for the kernel
        kernel = Kernel(self.as_symbolic)
        dimension = kernel.dimension
        kernel_keyvar = 'kernel_'+self.as_keyvar
        kernel_code_line = kernel_keyvar+' = '+kernel.as_code
        code += kernel_code_line.split('\n')

        # 2. Construction the NN Layer for the derivative .. or the keras line..
        # Todo

        # code += .. or  code.append(..)
        layer = MetaLayer(
            type_key="DerivativeLayer",
            name=self.as_keyvar,
            input_key=self.function.as_keyvar,
            output_key=self.as_keyvar,
            options= {
                'kernel_keyvar':kernel_keyvar,
                'size':kernel.size,
                'dimension':kernel.dimension}
        )
        skeleton = [layer]
        code += KerasCodingRule(skeleton).code

        return code



class MetaCoordinate(SuperMetaCoordinate):

    def __init__(self, coordinate, index=0):
        super().__init__(coordinate, index)
        self.input_shape = self._set_input_shape()

    def _set_input_shape(self):
        return 'input_shape_' + self.as_keyvar

class NNModelBuilder(ModelBuilder):

    _default_name = 'NNModel'
    from .templates.header_NNModelBuilder import template as template_header_NNModelBuilder
    from .templates.body_NNModelBuilder import template as template_body_NNModelBuilder
    from .templates.footer_NNModelBuilder import template as template_footer_NNModelBuilder

    template_header = template_header_NNModelBuilder
    template_body = template_body_NNModelBuilder # contains '__init__' function and NN model generator.
    template_footer = template_footer_NNModelBuilder

    def _create_meta_function(self, function, index):
        return MetaFunction(function, index)
    def _create_meta_prognostic_function(self, function, index):
        return MetaPrognosticFunction(function, index)
    def _create_meta_coordinate(self, coordinate, index):
        return MetaCoordinate(coordinate, index)
    def _create_meta_derivative(self, derivative):
        return MetaDerivative(derivative)

    def _make_trend_code(self):

        subs_trends = {Derivative(function.as_symbolic, self._time_symbol):function.trend
                        for function in self.prognostic_functions}

        trend_code = []
        for equation in self.pseudo_algebraic_system:
            # 1. Split equation
            lhs, rhs = equation.args

            # 2. Convert lhs into code rule
            lhs = lhs.subs(subs_trends)
            lhs = str(lhs)

            trend_code.append(f"#")
            trend_code.append(f"# Computation of {lhs}")
            trend_code.append(f"#")

            # 3. Compute NN Skeleton of the rhs
            code_skeleton = CodeSkeleton(rhs)
            skeleton = code_skeleton.skeleton

            if skeleton is not None:
                # 4. Update the output of the sekleton (last layer) by the lhs
                skeleton[-1].output_key = lhs
                # 5. Compute the keras code from the skeleton and add it to the trend_code
                trend_code += KerasCodingRule(code_skeleton.skeleton).code
            else:
                # No keras layer => lhs = rhs
                trend_code += [ lhs + "=keras.layers.Lambda(lambda x: x)(" + str(rhs) + ")" ]


        return trend_code

    @staticmethod
    def dx_coding_rule(coord):
        return f"self.dx[self.coordinates.index('{coord}')]"

