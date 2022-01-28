template = '''from pdenetgen.model import Model
import numpy as np
import tensorflow.keras as keras
from pdenetgen.symbolic.nn_builder import DerivativeFactory, TrainableScalarLayerFactory

class {{class_name}}(Model):

    # Prognostic functions (sympy functions):
    prognostic_functions = (
    {% for function in prognostic_functions %}        '{{function.as_keyvar}}',    # Write comments on the function here
    {% endfor %}    )

    {% if exogenous_functions_flag %}
    # Exogenous functions (sympy functions)
    exogenous_functions = (
    {% for function in exogenous_functions %}        '{{function.as_keyvar}}',    # Write comments on the exogenous function here
    {% endfor %}    )
    {% endif %}
    
    # Spatial coordinates
    coordinates = (
    {% for coord in coordinates %}        '{{coord.as_keyvar}}',    # Write comments on the coordinate here
    {% endfor %}    )

    {% if constant_functions_flag %}
    # Set constant functions
    constant_functions = (
    {% for function in constant_functions %}        '{{function.as_keyvar}}',    # Writes comment on the constant function here
    {% endfor %}    )
    {% endif %}

    {% if constants_flag %}
    # Set constants
    constants = (
    {% for constant in constants %}        '{{constant.as_keyvar}}',    # Writes comment on the constant here
    {% endfor %}    )
    {% endif %}

'''
