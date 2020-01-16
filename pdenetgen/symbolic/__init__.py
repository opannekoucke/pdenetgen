try:
    from .model_builder_NN import NNModelBuilder
    from .util import *
except ModuleNotFoundError as err:
    print(err)
    print("pdenetgen.symbolic not available")
