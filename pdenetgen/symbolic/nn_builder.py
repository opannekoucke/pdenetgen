from sympy import Mul, Add, Rational, Float, Integer, Pow, Function
from .util import ScalarSymbol, FunctionSymbol, TrainableScalar
from .tool import clean_latex_name
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class MetaLayer(object):
    """
    Token layer



    """
    def __init__(self,type_key=None, name=None,input_key=None,output_key=None, options=None):
        self.type_key = type_key # Type of layer: Add, Mul, Pow, Derivative, ..
        self.name = name # Name of the layer
        self.input_key = input_key # Input arguments
        self.output_key = output_key # Output arguments
        self.options = options # Option of the layer

    @staticmethod
    def from_layer(layer):
        MetaLayer(type_key=layer.type_key,
                  name=layer.name,
                  input_key=layer.input_key,
                  output_key=layer.output_key,
                  options=layer.options)
        return


class Counter(object):
    """ Handle the number of instance of a class """

    _cls_id = 0

    def __new__(cls, *args, **kwargs):
        instance = super(Counter, cls).__new__(cls)
        # Handle the counter
        instance._id = instance._cls_id
        instance._update_cls_id()
        return instance

    @classmethod
    def _update_cls_id(cls):
        cls._cls_id += 1

class Layer(Counter):
    """ Build a layer from the recursive exploration of the sub-expression

    Description
    -----------

    For any type of layer, this handle the iteration of layer used. For instance,
    when a layer is called its name is tag by an integer _id which is incremented
    after the call.

    """

    _name = None
    _output_key = None

    def __init__(self, expr):
        # Init layer properties
        self.expr = expr
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


class TrainableScalarLayer(Layer): # TrainableScalarMulLayer ?
    """
    Add a trainable scalar
    """
    _output_key = 'train_scalar'
    _name = 'TrainableScalar'
    
    def __init__(self, train_scalar, expr):
        super().__init__(expr)
        # Get options from train_scalar: If it is iterable: get the first, but consider all the labels
        # for instance if the product 'a*b' of trainable scalar 'a' and 'b' is encountered, the
        # this will produce a single layer with options of 'a' and label: 'a, b'
        if hasattr(train_scalar,'__iter__'):
            self.train_scalar = train_scalar[0]
            self.label = ', '.join([str(term) for term in train_scalar])
        else:
            self.train_scalar = train_scalar
            self.label = clean_latex_name(str(train_scalar))

    @property
    def name(self):
        if self._name is None:
            name = type(self).__name__
        else:
            name = self._name
        #return f"{name}_{self._id}"
        label = self.label.replace(',','_')
        return f"{name}_{label}"

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

        """
        meta_layer.options = {
            'init_value':self.train_scalar.init_value,
            'use_bias':self.train_scalar.use_bias,
            'label':self.label,
        }
        """
        # Add all options of train_scalar
        meta_layer.options = {option:getattr(self.train_scalar,option)
                                                    for option in TrainableScalar.options
        }
        # Add label
        meta_layer.options['label'] = self.label

        self._add_to_skeleton(meta_layer)

        return self.output_key, self._skeleton
    pass

class FunctionLayer(Layer):

    def __call__(self):
        output_key = str(self.expr)
        skeleton = None

        return output_key, skeleton


class LayerFactory(object):
    """ Translate a sympy expression into layers """

    def __new__(cls, expr):

        if isinstance(expr, Add):
            # Decompose the addition in sclar/vector addition
            # ---> use 'wild' for pattern research ? : not possible

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
            # ---> use 'wild' for pattern research ? : not possible

            # 1) check for decomposition (scalar * vector)
            scalar = 1
            trainable = False
            trainable_scalar = None
            other = []
            for arg in expr.args:
                if is_scalar(arg):
                    scalar *= arg
                elif isinstance(arg, TrainableScalar):
                    trainable = True
                    # retains only one trainable scalar (trainable scalars are absorbing: a*b => c not, (a,b) are
                    # not retained
                    trainable_scalar = arg if trainable_scalar is None else list(trainable_scalar)+[arg]
                else:
                    other.append(arg)
            other = Mul(*other)

            # 2) Applies 'Mul' for vector part and 'ScalarMul' for multiplication with a scalar.
            #   Three cases :
            #    i. saclar * scalar
            #            -> this situation should not appear if expressions are only with Float
            #   ii. trainable * vector
            #  iii. scalar * vector
            #   iv. vector * vector
            if trainable:
                #  ii. trainable * vector
                return TrainableScalarLayer(trainable_scalar, other)
            elif scalar == 1:
                # iv. vector*vector i.e. No multiplication with scalar
                return MulLayer(expr)
            else:
                # iii. Multiplication with scalar
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

        elif isinstance(expr, TrainableScalar):
            return TrainableScalarLayer(expr)

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
    """
    Translate MetaLayer into Keras code.
    """
    translate_to_keras = {
        'AddLayer': 'keras.layers.add',
        'MulLayer': 'keras.layers.multiply',
        'PowLayer': 'keras.layers.multiply',
        'DivLayer': 'keras.layers.Lambda',
        'ScalarPowLayer': 'keras.layers.Lambda',
        'ScalarAddLayer': 'keras.layers.Lambda',
        'ScalarMulLayer': 'keras.layers.Lambda',
        'DerivativeLayer': None, #'keras.layers.Conv{{dimension}}D',
        'TrainableScalarLayer': None, #'keras.layers.Conv{{dimension}}D',
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
                elif type_key == 'TrainableScalarLayer':
                    input_key=input_key[0]
                    label = meta_layer.options['label']
                    # Set init_options options
                    # 1. Set all options
                    init_value_options = [f"{option}={meta_layer.options[option]}" for option in TrainableScalar.options]
                    # 2. Convert init value options into string to insert in code line
                    init_value_options = ','.join(init_value_options)

                    new_lines = f"""{output_key} = TrainableScalarLayerFactory(input_shape={input_key}.shape, name='{name}', 
                        {init_value_options})({input_key})
                        #TrainableScalar name: '{label}' """
                else:
                    raise NotImplementedError

                code.append(new_lines)

            self._code = code

        return self._code

class TrainableScalarLayerFactory(object):
    """
    Build a trainable scalar layer

    Description
    -----------

    The implementation of trainable scalar layer relies on convolutional neural network.
    A trainable scalar layer corresponds to the layer $a f(t,x)$ where $a$ is unknown and $f(t,x)$ is a field.
    in this case, a Conv1D is used (t being a temporal coordinate):

    $a f(t,x)$  becomes a Conv1D layer **without bias** where with a kernel size of $1$.

    Generate a trainable scalar layer depending on the input shape to switch between Conv1D, Conv2D and Conv3D
    """

    def __new__(cls, input_shape, init_value=None, name=None, use_bias=False,
                            mean=0., stddev=1., wl2=None, seed=None,
                            **kwargs):
        """

        :param input_shape: shape of the input of the layer.
        :param init_value: it is possible to set the value of the scalar to train.
        :param name:
        :param use_bias: No bias should be used
        :param mean: mean of the random normal law sample (0. is often used)
        :param stddev: std dev of the random normal law sample (0.05 is often used)
        :param seed:   seed of the random sample (None means the seed is ?)
        :param wl2:   regularizaiton (0.001 isoften used)
        :param kwargs:
        """
        # 1. Set the dimension from input_shape information
        dimension = len(input_shape[1:-1])

        # 2. Set kernel considering the boundary condition.
        options = {
                    'padding':'same',
                    'activation':'linear',
                    'use_bias':use_bias,
                    'name':name,
                    'trainable': True,
        }

        if wl2 is not None:
            options['kernel_regularizer'] = keras.regularizers.l2(wl2)
        if init_value is None:
            options['kernel_initializer'] = keras.initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)
        else:
            options['weights'] = [np.array(init_value).reshape(dimension*(1,)+(1,1))]

        kernel_size = dimension*(1,)
        nfilter = 1

        if dimension == 1:
            return keras.layers.Conv1D(nfilter, kernel_size, **options)

        elif dimension == 2:
            return keras.layers.Conv2D(nfilter, kernel_size, **options)

        elif dimension == 3:
            return keras.layers.Conv3D(nfilter, kernel_size, **options)

        else:
            raise NotImplementedError


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

    def __new__(cls, kernel_size,
                            kernel=None,
                            name=None,
                            periodic=True,
                            nfilter=1,
                            activation='linear',
                            use_bias=False,
                            wl2 = 0.001,
                            mean = 0.0,
                            stddev = 0.05,
                            seed = None,
                            #boundary_condition='periodic',
                            **kwargs):
        """
        Create Derivative layer
        :param kernel_size:
        :param kernel:
                    * when is None:
                        the kernel can be deduced from learning
                    * when provided:
                        the kernel should corresponds to finite difference stencil of the derivative
        :param name: string for name of layer
        :param periodic: boolean to apply boundary condition
        :param activation: activation layer used 'linear' or 'relu' or ..
        :param use_bias: boolean for using (or not) a bias, no bias is used if kernel is specified.
        :param nfilter: number of output channels
        :return:
        """

        """
        ..todo::
          [ ] replace 'periodic' by 'boundary_condition' with options: 'periodic', 'dirichlet', 'neumann',.. 
        """

        # 1. Disable periodic boundary if support of kernel_size is 1 as in (1,1,1) 
        if np.prod(kernel_size)==1:
            periodic = False

        # 1. Handle boundary conditions to create appropriate derivative
        # 1.1 Periodic boundary condition
        if periodic: #or boundary_condition=='periodic':
            def PeriodicDerivative(conv):
                periodized = PeriodicFactory(kernel_size)(conv)
                derivative = DerivativeFactory(kernel_size,
                                               kernel = kernel,
                                               name = name,
                                               periodic = False,
                                               nfilter = nfilter,
                                               activation = activation,
                                               use_bias = use_bias,
                                               wl2 = wl2,
                                               mean = mean,
                                               stddev = stddev,
                                               seed = seed,
                                               **kwargs)(periodized)
                cropped = CropFactory(kernel_size)(derivative)
                return cropped

            return PeriodicDerivative

        # 2. Set kernel considering the boundary condition.
        options = {
                    'padding':'same',  # should depends on boundary condition.. this should be modified later.
                    'activation':activation,
                    'use_bias':use_bias,
                    'name':name,
        }
        if kernel is not None:
            #print('Set kernel information')
            options['weights'] = [kernel]
            options['trainable'] = False
            options['use_bias'] = False # no bias is used if kernel is specified
        else:
            #print('Kernels in derivative are unknown and set to trainable')
            print(f'randomized kernel with l2 regularization (wl2: {wl2}, mean: {mean}, stddev: {stddev} )')
            if wl2 is not None:
                options['kernel_regularizer'] = keras.regularizers.l2(wl2)
            options['kernel_initializer'] = keras.initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)

        dimension = len(kernel_size)
        if dimension == 1:
            return keras.layers.Conv1D(nfilter, kernel_size, **options)

        elif dimension == 2:
            return keras.layers.Conv2D(nfilter, kernel_size, **options)

