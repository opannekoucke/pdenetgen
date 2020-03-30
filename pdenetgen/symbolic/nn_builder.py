from sympy import Mul, Add, Rational, Float, Integer, Pow, Function
from .util import ScalarSymbol, FunctionSymbol
import keras
import tensorflow as tf
import numpy as np

class MetaLayer(object):

    def __init__(self,type_key=None, name=None,input_key=None,output_key=None,options=None):
        self.type_key = type_key
        self.name = name
        self.input_key = input_key
        self.output_key = output_key
        self.options = options

    @staticmethod
    def from_layer(layer):
        MetaLayer(type_key=layer.type_key,
                  name=layer.name,
                  input_key=layer.input_key,
                  output_key=layer.output_key,
                  options=layer.options)
        return

class Layer(object):
    """ Build a layer from the recursive exploration of the sub-expression """

    _id = 0
    _name = None
    _output_key = None

    def __init__(self, expr):
        self.expr = expr
        self.number = self._id
        self._update_id()
        self._skeleton = []
        self._input_key = []
        self._options = None

    def _agregate_sub_expressions(self):
        # 1) Agregates code
        for arg in self.expr.args:
            output_key, sub_skeleton = LayerFactory(arg)()
            self.add_input_key(output_key)
            self._add_to_skeleton(sub_skeleton)

    def _extract_options(self):
        pass

    @property
    def as_MetaLayer(self):

        meta_layer = MetaLayer(
                    type_key=self.type_key,
                    name=self.name,
                    input_key=self.input_key,
                    output_key=self.output_key,
                    options=self.options
        )
        return meta_layer

    def add_input_key(self, input_key):
        if isinstance(input_key,list):
            self._input_key += input_key
        else:
            self._input_key.append(input_key)

    @property
    def type_key(self):
        return type(self).__name__

    @property
    def input_key(self):
        return self._input_key

    @property
    def options(self):
        return self._options

    def __call__(self):
        """ Construct the skeleton of a symbolic expression """

        """ 
        Code description:

           1) Feed self._skeleton and self._input_key from sub-expressions
           2) Extract option when needed (Derivative,.. )
           3) Feed self._skeleton with the present layer
        """
        # 1) Feed from sub-expressions: this modifies self._skeleton & self._input_key
        self._agregate_sub_expressions()

        # 2) Extract options
        self._extract_options()

        # 3) Add layer to skeleton
        self._add_to_skeleton(self.as_MetaLayer)

        return self.output_key, self._skeleton

    @classmethod
    def _update_id(cls):
        cls._id += 1

    def _add_to_skeleton(self, skeleton):
        if skeleton is not None:
            if isinstance(skeleton, list):
                self._skeleton += skeleton
            else:
                self._skeleton.append(skeleton)

    @property
    def name(self):
        if self._name is None:
            name = type(self).__name__
        else:
            name = self._name
        return f"{name}_{self._id}"

    @property
    def output_key(self):
        if self._output_key is None:
            return self.name
        else:
            return f"{self._output_key}_{self._id}"


class AddLayer(Layer):
    _output_key = 'add'
    pass


class MulLayer(Layer):
    _output_key = 'mul'
    pass



class ScalarAddLayer(Layer):
    _output_key = 'sc_add'

    def __init__(self, scalar, expr):
        super().__init__(expr)
        self.scalar = ScalarLayer(scalar)()[0]

    def __call__(self):
        """ Construct the skeleton of a symbolic expression """

        """ 
        Code description:

           1) Feed self._skeleton and self._input_key from sub-expressions
           2) Extract option when needed (Derivative,.. )
           3) Feed self._skeleton with the present layer
        """

        # 1) Feed from sub-expressions: this modifies self._skeleton & self._input_key
        ouptut_key, skeleton = LayerFactory(self.expr)()
        self.add_input_key(ouptut_key)
        self._add_to_skeleton(skeleton)

        # 3) Add layer to skeleton
        meta_layer = self.as_MetaLayer
        meta_layer.input_key = [self.scalar] + self.input_key
        self._add_to_skeleton(meta_layer)

        return self.output_key, self._skeleton


class ScalarMulLayer(Layer):
    _output_key = 'sc_mul'

    def __init__(self, scalar, expr):
        super().__init__(expr)
        self.scalar = ScalarLayer(scalar)()[0]

    def __call__(self):
        """ Construct the skeleton of a symbolic expression """

        """ 
        Code description:

           1) Feed self._skeleton and self._input_key from sub-expressions
           2) Extract option when needed (Derivative,.. )
           3) Feed self._skeleton with the present layer
        """
        # 1) Feed from sub-expressions: this modifies self._skeleton & self._input_key
        output_key, skeleton = LayerFactory(self.expr)()
        self.add_input_key(output_key)
        self._add_to_skeleton(skeleton)

        # 3) Add layer to skeleton
        meta_layer = self.as_MetaLayer
        meta_layer.input_key = [self.scalar] + self.input_key
        self._add_to_skeleton(meta_layer)

        return self.output_key, self._skeleton

class DivLayer(Layer):
    _output_key = 'div'

    def __call__(self):
        """ Construct the skeleton of a symbolic expression """

        """ 
        Code description:

           1) Feed self._skeleton and self._input_key from sub-expressions
           2) Extract option when needed (Derivative,.. )
           3) Feed self._skeleton with the present layer
        """
        # 1) Feed from sub-expressions: this modifies self._skeleton & self._input_key
        output_key, skeleton = LayerFactory(self.expr)()
        self._add_to_skeleton(skeleton)

        # 3) Add layer to skeleton
        meta_layer = self.as_MetaLayer
        meta_layer.input_key = output_key
        self._add_to_skeleton(meta_layer)

        return self.output_key, self._skeleton

class PowLayer(Layer):
    """ Pow layer for x**a where 'a' is an integer """

    _output_key = 'pow'

    def __init__(self, scalar, expr):
        if isinstance(scalar, Integer):
            if scalar<0:
                self.inverse = True
                scalar = -scalar
            else:
                self.inverse = False
            self.scalar = str(scalar)
        else:
            raise ValueError(f"exponent {scalar} should be an Integer")
        super().__init__(expr)


    def __call__(self):
        """ Construct the skeleton of a symbolic expression """

        """ 
        Code description:

           1) Feed self._skeleton and self._input_key from sub-expressions
           2) Extract option when needed (Derivative,.. )
           3) Feed self._skeleton with the present layer
        """
        # 1) Feed from sub-expressions: this modifies self._skeleton & self._input_key
        if self.inverse:
            output_key, skeleton = DivLayer(self.expr)()
        else:
            output_key, skeleton = LayerFactory(self.expr)()

        self.add_input_key(output_key)
        self._add_to_skeleton(skeleton)

        # 3) Add layer to skeleton
        meta_layer = self.as_MetaLayer
        meta_layer.input_key = [self.scalar] + self.input_key
        self._add_to_skeleton(meta_layer)

        return self.output_key, self._skeleton


class ScalarPowLayer(Layer):
    """ Pow layer for x**a where 'a' is not an integer """
    _output_key = 'spow'

    def __init__(self, scalar, expr):
        super().__init__(expr)
        self.scalar = str(float(Float(scalar)))

    def __call__(self):
        """ Construct the skeleton of a symbolic expression """

        """ 
        Code description:

           1) Feed self._skeleton and self._input_key from sub-expressions
           2) Extract option when needed (Derivative,.. )
           3) Feed self._skeleton with the present layer
        """
        # 1) Feed from sub-expressions: this modifies self._skeleton & self._input_key
        output_key, skeleton = LayerFactory(self.expr)()
        self.add_input_key(output_key)
        self._add_to_skeleton(skeleton)

        # 3) Add layer to skeleton
        meta_layer = self.as_MetaLayer
        meta_layer.input_key = [self.scalar] + self.input_key
        self._add_to_skeleton(meta_layer)

        return self.output_key, self._skeleton

class DerivativeLayer(Layer):
    """
     - get the derivative order (partial order)
     - compute the stencil from sympy diff operator
     - define the kernel
    """
    _output_key = 'deriv'

    def _extract_options(self):
        raise NotImplementedError

def is_scalar(expr):
    if isinstance(expr, (Integer, Float, Rational, ScalarSymbol, int, float)):
        return True
    elif isinstance(expr, (Function, Add, Mul, Pow)):
        return sum([is_scalar(arg) for arg in expr.args]) == len(expr.args)
    else:
        return False

class ScalarLayer(Layer):

    def __call__(self):

        if not is_scalar(self.expr):
            raise ValueError(f"{self.expr} is not a scalar")

        expr = self.expr
        # Replace Rationals by Float
        expr = expr.subs({scalar: Float(scalar) for scalar in expr.atoms(Rational)})
        # Replace Integer by Float
        expr = expr.subs({scalar: Float(scalar) for scalar in expr.atoms(Integer)})

        output_key = str(expr)

        # Replace long string for float
        for scalar in expr.atoms(Float):
            output_key = output_key.replace(str(scalar), str(float(scalar)))

        skeleton = None

        return output_key, skeleton

class FunctionLayer(Layer):

    def __call__(self):
        output_key = str(self.expr)
        skeleton = None

        return output_key, skeleton

class LayerFactory(object):

    def __new__(cls, expr):

        if isinstance(expr, Add):
            # Decompose the addition in sclar/vector addition
            # ---> use 'wild' for pattern research.

            # 1) check for decomposition (scalar + vector)
            scalar = 0
            other = 0
            for arg in expr.args:
                if is_scalar(arg):
                    scalar += arg

                else:
                    other += arg

            # 2) Applies 'Add' for vector part and 'ScalarAdd' for addition with scalar.
            #   Three cases :
            #    i. saclar + scalar
            #            -> this situation should not appear if expressions are only with Float
            #   ii. scalar + vector
            #  iii. vector + vector
            if scalar == 0:
                # iii. No addition with scalar
                return AddLayer(expr)
            else:
                # ii. Addition with scalar
                return ScalarAddLayer(scalar, other)

        elif isinstance(expr, Mul):
            # Decompose the multiplication in sclar/vector multiplication
            # ---> use 'wild' for pattern research.

            # 1) check for decomposition (scalar + vector)

            # 2) Applies 'Mul' for vector part and 'ScalarMul' for addition with scalar.
            #   Three cases :
            #    i. saclar * scalar
            #            -> this situation should not appear if expressions are only with Float
            #   ii. scalar * vector
            #  iii. vector * vector
            # 1) check for decomposition (scalar + vector)
            scalar = 1
            other = []
            for arg in expr.args:
                if is_scalar(arg):
                    scalar *= arg
                else:
                    other.append(arg)
            other = Mul(*other)

            # 2) Applies 'Add' for vector part and 'ScalarAdd' for addition with scalar.
            #   Three cases :
            #    i. saclar + scalar
            #            -> this situation should not appear if expressions are only with Float
            #   ii. scalar + vector
            #  iii. vector + vector
            if scalar == 1:
                # iii. No addition with scalar
                return MulLayer(expr)
            else:
                # ii. Addition with scalar
                return ScalarMulLayer(scalar, other)

        elif isinstance(expr, Pow):
            sub_expr, exponent = expr.args
            if isinstance(exponent, Integer):
                if exponent == Integer(-1):
                    return DivLayer(sub_expr)
                else:
                    return PowLayer(exponent, sub_expr)

            elif isinstance(exponent, (Float, Rational)):
                return ScalarPowLayer(exponent, sub_expr)

            elif isinstance(exponent, ScalarSymbol) and exponent.is_integer:
                return PowLayer(exponent, sub_expr)

            elif isinstance(exponent, ScalarSymbol) and exponent.is_real:
                return ScalarPowLayer(exponent, sub_expr)

            else:
                raise ValueError(f"{exponent} is not Integer/Float/Rational/Symbol -- integer or real --")

        elif isinstance(expr, FunctionSymbol):
            return FunctionLayer(expr)

        elif isinstance(expr, (Float, Integer, Rational, ScalarSymbol)):
            return ScalarLayer(expr)

        else:
            raise NotImplementedError

class CodeSkeleton(object):

    def __init__(self, expr):
        self._expr = expr
        output_key, skeleton = LayerFactory(self._expr)()
        self._output_key = output_key
        self._skeleton = skeleton

    @property
    def skeleton(self):
        return self._skeleton

    @property
    def output_key(self):
        return self._output_key

    @property
    def expr(self):
        return self._expr


class KerasCodingRule(object):
    translate_to_keras = {
        'AddLayer': 'keras.layers.add',
        'MulLayer': 'keras.layers.multiply',
        'PowLayer': 'keras.layers.multiply',
        'DivLayer': 'keras.layers.Lambda',
        'ScalarPowLayer': 'keras.layers.Lambda',
        'ScalarAddLayer': 'keras.layers.Lambda',
        'ScalarMulLayer': 'keras.layers.Lambda',
        'DerivativeLayer': 'keras.layers.Conv{{dimension}}D',
    }

    def __init__(self, skeleton):
        self.skeleton = skeleton
        self._code = None

    @property
    def code(self):
        if self._code is None:

            code = []
            for meta_layer in self.skeleton:

                output_key = meta_layer.output_key
                input_key = meta_layer.input_key
                name = meta_layer.name
                type_key = meta_layer.type_key
                function = self.translate_to_keras[type_key]

                if type_key in ['AddLayer', 'MulLayer']:
                    input_args = '[' + ",".join(input_key) + ']'
                    new_lines = f"{output_key} = {function}({input_args},name='{name}')"
                elif type_key == 'PowLayer':
                    # Convert "x**a" by  an 'a' times product "x*x ... *x"
                    exponent,x  = input_key
                    exponent = int(exponent)
                    x_comma = x+','
                    input_args = '[' + exponent *x_comma  + ']'
                    new_lines = f"{output_key} = {function}({input_args} ,name='{name}')"
                elif type_key == 'ScalarPowLayer':
                    exponent,x  = input_key
                    new_lines = f"{output_key} = {function}(lambda x: x**{exponent},name='{name}')({x})"
                elif type_key == 'DivLayer':
                    new_lines = f"{output_key} = {function}(lambda x: 1/x,name='{name}')({input_key})"
                elif type_key == 'ScalarAddLayer':
                    scalar, other = input_key
                    new_lines = f"{output_key} = {function}(lambda x: x+{scalar},name='{name}')({other})"
                elif type_key == 'ScalarMulLayer':
                    scalar, other = input_key
                    new_lines = f"{output_key} = {function}(lambda x: {scalar}*x,name='{name}')({other})"
                elif type_key == 'DerivativeLayer':
                    kernel_keyvar = meta_layer.options['kernel_keyvar']
                    kernel_size = meta_layer.options['size']
                    #new_lines = f"{output_key} = {function.replace('{{dimension}}',str(dimension))}(1,{kernel_size},weights=[{kernel_keyvar}],trainable=False,padding='same',activation='linear',use_bias=False,name='{name}')({input_key})"
                    new_lines = f"{output_key} = DerivativeFactory({kernel_size},kernel={kernel_keyvar},name='{name}')({input_key})"
                else:
                    raise NotImplementedError

                code.append(new_lines)

            self._code = code

        return self._code


class PeriodicFactory(object):

    def __new__(cls, kernel_size, **kwargs):

        dimension = len(kernel_size)

        if dimension == 1:
            return Periodic1D(kernel_size, **kwargs)
        elif dimension == 2:
            return Periodic2D(kernel_size, **kwargs)
        else:
            raise NotImplementedError('Periodic boundary for dimension higher than 2 is not implemented')


class Periodic(keras.layers.Layer):
    """ Periodization of the grid in accordance with filter size"""

    def __init__(self, kernel_size, **kwargs):
        # Convert network into [:,network]
        self.kernel_size = kernel_size
        if (np.array(kernel_size) % 2 == len(kernel_size) * (1,)).all():
            self.add_columns = np.array(kernel_size, dtype=int) // 2
        else:
            raise ValueError(f"kernel_size {kernel_size} is not **odd** integer")
        super().__init__(**kwargs)

    def call(self, x):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        raise NotImplementedError

    def get_config(self):
        config = {
            'add_columns': self.add_columns,
            'kernel_size': self.kernel_size
        }
        # base_config = super(Layer, self).get_config()
        # return dict(list(base_config.items()) + list(config.items()))
        return config


class Periodic1D(Periodic):
    """ Periodization of the grid in accordance with filter size"""

    def call(self, x):
        """ Add self.kernel_size[-1] columns in the geographical domain """
        # x[batch, x_shape, channels]
        left_columns = x[:, :self.add_columns[0], :]
        right_columns = x[:, -self.add_columns[0]:, :]
        periodic_x = tf.concat([right_columns, x, left_columns], axis=1)
        return periodic_x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] += 2 * self.add_columns[0]
        return tuple(output_shape)


class Periodic2D(Periodic):
    """ Periodization of the grid in accordance with filter size"""

    def call(self, x):
        """ Add self.kernel_size[-1] columns in the geographical domain """
        # x[batch, x_shape, y_shape, channels]
        left_columns = x[:, :self.add_columns[0], :, :]
        right_columns = x[:, -self.add_columns[0]:, :, :]
        periodic_x = tf.concat([right_columns, x, left_columns], axis=1)
        left_columns = periodic_x[:, :, :self.add_columns[1], :]
        right_columns = periodic_x[:, :, -self.add_columns[1]:, :]
        periodic_x = tf.concat([right_columns, periodic_x, left_columns], axis=2)
        return periodic_x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] += 2 * self.add_columns[0]
        output_shape[2] += 2 * self.add_columns[1]
        return tuple(output_shape)


class CropFactory(object):

    def __new__(cls, kernel_size, **kwargs):

        dimension = len(kernel_size)

        if dimension == 1:
            return Crop1D(kernel_size, **kwargs)
        elif dimension == 2:
            return Crop2D(kernel_size, **kwargs)
        else:
            raise NotImplementedError('Crop for dimension higher than 2 is not implemented')


class Crop(keras.layers.Layer):

    def __init__(self, kernel_size, **kwargs):
        self.kernel_size = kernel_size
        if (np.array(kernel_size) % 2 == len(kernel_size) * (1,)).all():
            self.suppress_columns = np.array(kernel_size, dtype=int) // 2
        else:
            raise ValueError(f"kernel_size {kernel_size} is not **odd** integer")
        super().__init__(**kwargs)

    def get_config(self):
        config = {
            'suppress_columns': self.suppress_columns,
            'kernel_size': self.kernel_size
        }
        # base_config = super(Layer, self).get_config()
        # return dict(list(base_config.items()) + list(config.items()))
        return config


class Crop2D(Crop):
    """ Extract center of the domain """

    def call(self, x):
        """ Add self.kernel_size[-1] columns in the geographical domain """
        out = x[:, :, self.suppress_columns[1]:-self.suppress_columns[1], :]  # eliminate additionnal boundaries
        return out[:, self.suppress_columns[0]:-self.suppress_columns[0], :, :]

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] -= 2 * self.suppress_columns[0]
        output_shape[2] -= 2 * self.suppress_columns[1]
        return tuple(output_shape)


class Crop1D(Crop):
    """ Spherical periodization of the grid in accordance with filter size"""

    def call(self, x):
        """ Add self.kernel_size[-1] columns in the geographical domain """
        #out = x[:, self.suppress_columns[0]:-self.suppress_columns[0], :]
        #return tf.cast(out,dtype='float32')  # eliminate additionnal boundaries
        return x[:, self.suppress_columns[0]:-self.suppress_columns[0], :]

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] -= 2 * self.suppress_columns[0]
        return tuple(output_shape)


class DerivativeFactory(object):

    def __new__(cls, kernel_size, kernel=None, name=None, periodic=True, nfilter=1):
        """
        Create Derivative layer
        :param kernel_size:
        :param kernel:
                    * when is None:
                        the kernel can be deduced from learning
                    * when provided:
                        the kernel should corresponds to finite difference stencil of the derivative
        :param name:
        :param periodic:
        :param nfilter:
        :return:
        """

        dimension = len(kernel_size)

        if periodic:
            def PeriodicDerivative(conv):
                periodized = PeriodicFactory(kernel_size)(conv)
                derivative = DerivativeFactory(kernel_size, kernel, name, periodic=False, nfilter=nfilter)(periodized)
                cropped = CropFactory(kernel_size)(derivative)
                return cropped

            return PeriodicDerivative

        options = {
                    'padding':'same',
                    'activation':'linear',
                    'use_bias':False,
                    'name':name,
        }
        if kernel is not None:
            options['weights'] = [kernel]
            options['trainable'] = False
        else:
            print(f'Kernel for derivative `{name}` is set ** trainable **')
            #wl2 = 0.001
            #options['kernel_regularizer'] = keras.regularizers.l2(wl2)
            #options['kernel_initializer'] = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

        if dimension == 1:
            return keras.layers.Conv1D(nfilter, kernel_size, **options)

        elif dimension == 2:
            return keras.layers.Conv2D(nfilter, kernel_size, **options)

