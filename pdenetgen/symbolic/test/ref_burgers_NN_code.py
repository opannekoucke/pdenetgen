ref_burgers_NN_code = '''from pdenetgen.model import Model
import numpy as np
import tensorflow.keras as keras
from pdenetgen.symbolic.nn_builder import DerivativeFactory, TrainableScalarLayerFactory

class Burgers(Model):

    # Prognostic functions (sympy functions):
    prognostic_functions = (
            'u',    # Write comments on the function here
        )

    
    
    # Spatial coordinates
    coordinates = (
            'x',    # Write comments on the coordinate here
        )

    

    
    # Set constants
    constants = (
            'kappa',    # Writes comment on the constant here
        )
    

    def __init__(self, shape=None, lengths=None, **kwargs):

        super().__init__() # Time scheme is set from Model.__init__()
                
        #---------------------------------
        # Set index array from coordinates
        #---------------------------------
        
        # a) Set shape
        shape = len(self.coordinates)*(100,) if shape is None else shape 
        if len(shape)!=len(self.coordinates):
            raise ValueError(f"len(shape) {len(shape)} is different from len(coordinates) {len(self.coordinates)}")
        else:
            self.shape = shape
    
        # b) Set input shape for coordinates
        self.input_shape_x = shape[0]
        
                
        # c) Set lengths
        lengths = len(self.coordinates)*(1.0,) if lengths is None else lengths
        if len(lengths)!=len(self.coordinates):
            raise ValueError(f"len(lengths) {len(lengths)} is different from len(coordinates) {len(self.coordinates)}")        
        else:
            self.lengths = lengths
            
        # d) Set indexes
        self._index = {}
        for k,coord in enumerate(self.coordinates):
            self._index[(coord,0)] = np.arange(self.shape[k], dtype=int)            
        
        # Set x/dx
        #-------------
        self.dx = tuple(length/shape for length, shape in zip(self.lengths, self.shape))
        self.x = tuple(self.index(coord,0)*dx for coord, dx in zip(self.coordinates, self.dx))
        self.X = np.meshgrid(*self.x)

        

        
        #---------------------------
        # Set constants of the model
        #---------------------------
          
        # Set a default nan value for constants
        self.kappa = np.nan # @@ set constant value @@
        
                
        # Set constant values from external **kwargs (when provided)
        for key in kwargs:
            if key in self.constants:
                setattr(self, key, kwargs[key])
        
        # Alert when a constant is np.nan
        for constant in self.constants:
            if getattr(self, constant) is np.nan:
                print(f"Warning: constant `{constant}` has to be set")
        
        
        # Set NN models
        self._trend_model = None
        self._exogenous_model = None
    
    def index(self, coord, step:int):
        """ Return int array of shift index associated with coordinate `coord` for shift `step` """
        # In this implementation, indexes are memory saved in a dictionary, feed at runtime 
        if (coord,step) not in self._index:
            self._index[(coord,step)] = (self._index[(coord,0)]+step)%self.shape[self.coordinates.index(coord)]
        return self._index[(coord,step)] 
    
    def _make_trend_model(self):
        """ Generate the NN used to compute the trend of the dynamics """
                        
        # Alias for constants
        #--------------------
        kappa = self.kappa
        if kappa is np.nan:
            raise ValueError("Constant 'kappa' is not set")
                 
                
        
        # Set input layers
        #------------------
        # Set Alias for coordinate input shapes
        input_shape_x = self.input_shape_x
                
        # Set input shape for prognostic functions
        u = keras.layers.Input(shape =(input_shape_t,input_shape_x,1,))
         
                
        
         
        # Keras code
         
        # 2) Implementation of derivative as ConvNet
        # Compute derivative
        #-----------------------
        #
        #  Warning: might be modified to fit appropriate boundary conditions. 
        #
        kernel_Du_x_o2 = np.asarray([self.dx[self.coordinates.index('x')]**(-2),
        -2/self.dx[self.coordinates.index('x')]**2,
        self.dx[self.coordinates.index('x')]**(-2)]).reshape((3,)+(1,1))
        Du_x_o2 = DerivativeFactory((3,),kernel=kernel_Du_x_o2,name='Du_x_o2')(u)
        
        kernel_Du_x_o1 = np.asarray([-1/(2*self.dx[self.coordinates.index('x')]),0.0,
        1/(2*self.dx[self.coordinates.index('x')])]).reshape((3,)+(1,1))
        Du_x_o1 = DerivativeFactory((3,),kernel=kernel_Du_x_o1,name='Du_x_o1')(u)
        
                
        
        # 3) Implementation of the trend as NNet
        
        #
        # Computation of trend_u
        #
        sc_mul_0 = keras.layers.Lambda(lambda x: kappa*x,name='ScalarMulLayer_0')(Du_x_o2)
        mul_0 = keras.layers.multiply([Du_x_o1,u],name='MulLayer_0')
        sc_mul_1 = keras.layers.Lambda(lambda x: -1.0*x,name='ScalarMulLayer_1')(mul_0)
        trend_u = keras.layers.add([sc_mul_0,sc_mul_1],name='AddLayer_0')
        
        
        # 4) Set 'input' of model
        inputs = [
                # Prognostic functions
                u,
                
            ]         
   
        # 5) Set 'outputs' of model 
        outputs = [
            trend_u,
            ]     
        
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        #model.trainable = False
        self._trend_model = model
        
    def trend(self, t, state):
        """ Trend of the dynamics """
        
        if self._trend_model is None:
            self._make_trend_model()

        # Init output state with pointer on data
        #-------------------------------------------

        #   a) Set the output array
        dstate = np.zeros(state.shape)

        #   b) Set pointers on output array `dstate` for the computation of the physical trend (alias only).
        du = dstate[0]
        

        # Load physical functions from state
        #------------------------------------
        u = state[0]
                  
         
        
        
        # Compute the trend value from model.predict
        #-------------------------------------------  
        inputs = [
            # Prognostic functions
                u,
                ]
        
        dstate = self._trend_model.predict( inputs )
        
        if not isinstance(dstate,list):
            dstate = [dstate]
        
        return np.array(dstate)
        
    
    def _make_dynamical_trend(self):
        """
        Computation of a trend model so to be used in a time scheme (as solving a dynamical system or an ODE)
        
        Description:
        ------------
        
        In the present implementation, the inputs of the trend `self._trend_model` is a list of fields, while
        entry of a time-scheme is a single array which contains all fields.
        
        The aims of `self._dynamical_trend` is to produce a Keras model which:
        1. takes a single array as input
        2. extract the `self._trend_model` input list from the input array
        3. compute the trends from `self._trend_model`
        4. outputs the trends as a single array
        
        Explaination of the code:
        -------------------------
        
        Should implement a code as the following, that is valid for the PKF-Burgers         
                
        def _make_dynamical_trend(self):

            if self._trend_model is None:
                self._make_trend_model()

            # 1. Extract the input of the model
            
            # 1.1 Set the input as an array
            state = keras.layers.Input(shape=(3,self.input_shape_x,1))

            # 1.2 Extract each components of the state
            u = keras.layers.Lambda(lambda x : x[:,0,:,:])(state)
            V = keras.layers.Lambda(lambda x : x[:,1,:,:])(state)
            nu_u_xx = keras.layers.Lambda(lambda x : x[:,2,:,:])(state)

            # 2. Compute the trend
            trend_u, trend_V, trend_nu = self._trend_model([u,V,nu_u_xx])
            
            # 3. Outputs the trend as a single array

            # 3.1 Reshape trends
            trend_u = keras.layers.Reshape((1,self.input_shape_x,1))(trend_u)
            trend_V = keras.layers.Reshape((1,self.input_shape_x,1))(trend_V)
            trend_nu = keras.layers.Reshape((1,self.input_shape_x,1))(trend_nu)
                        
            # 3.2 Concatenates all trends
            trends = keras.layers.Concatenate(axis=1)([trend_u,trend_V,trend_nu])
            
            # 4. Set the dynamical_trend model
            self._dynamical_trend = keras.models.Model(inputs=state,outputs=trends)        
        """
        
        
        if self._trend_model is None:            
            self._make_trend_model()
            
            
        for exclude_case in ['constant_functions','exogenous_functions']:
            if hasattr(self,exclude_case):
                raise NotImplementedError(f'Design of dynamical_model with {exclude_case} is not implemented')

                
        # Case 1 --  corresponds to the _trend_model if input is a single field                        
        if not isinstance(self._trend_model.input_shape, list):
            self._dynamical_trend = self._trend_model
            return

        # Case 2 -- Case where multiple list is used
        
        # 1. Extract the input of the model

        # 1.1 Set the input as an array
        """ from PKF-Burgers code:
        state = keras.layers.Input(shape=(3,self.input_shape_x,1))
        """
        
        # 1.1.1 Compute the input_shape from _trend_model
        shapes = []
        dimensions  = []
        for shape in self._trend_model.input_shape:
            shape = shape[1:] # Exclude batch_size (assumed to be at first)
            shapes.append(shape)
            dimensions.append(len(shape)-1)
            
        max_dimension = max(dimensions)
        if max_dimension!=1:
            if 1 in dimensions:
                raise NotImplementedError('1D fields incompatible with 2D/3D fields')            

        # todo: add test to check compatibility of shapes!!!!
        
        if max_dimension in [1,2]:
            input_shape = (len(shapes),)+shapes[0]
        elif max_dimension==3:
            
            # a. check the size of 2D fields: this is given by the first 2D field.
            for shape, dimension in zip(shapes, dimensions):
                if dimension==2:
                    input_shape_2D = shape
                    break
                    
            # b. Compute the numbers of 2D fields: this corresponds to the number of 3D layers and the number of 2D fields.
        
            for shape, dimension in zip(shapes, dimensions):
                if dimension==2:
                    nb_outputs += 1
                else:
                    nb_outputs += shape[0]
                    
            input_shape = (nb_outputs,)+input_shape_2D
            
        # 1.1.2 Init the state of the dynamical_trend
        state = keras.layers.Input(shape=input_shape)
        
        # 1.2 Extract each components of the state
        """ From PKF-Burgers code:
        
        u = keras.layers.Lambda(lambda x : x[:,0,:,:])(state)
        V = keras.layers.Lambda(lambda x : x[:,1,:,:])(state)
        nu_u_xx = keras.layers.Lambda(lambda x : x[:,2,:,:])(state)
        
        inputs = [u, V, nu_u_xx]
        """

        def get_slice(dimension, k):
            def func(x):
                if dimension == 1:
                    return x[:,k,:,:]
                elif dimension == 2:
                    return x[:,k,:,:]
            return func
            
        def get_slice_3d(start,end):
            def func(x):
                return x[:,start:end,:,:,:]
            return func

        inputs = []
        if max_dimension in [1,2]:
            for k in range(len(shapes)):
                inputs.append(keras.layers.Lambda(get_slice(max_dimension,k))(state))
                #if max_dimension==1:
                #    inputs.append(keras.layers.Lambda(lambda x : x[:,k,:,:])(state))
                #
                #if max_dimension==2:
                #    inputs.append(keras.layers.Lambda(lambda x : x[:,k,:,:,:])(state))                    
        else:
            k=0
            for shape, dimension in zip(shapes, dimensions):
                if dimension==2:
                    #inputs.append(keras.layers.Lambda(lambda x : x[:,k,:,:,:])(state))
                    inputs.append(keras.layers.Lambda(get_slice(dimension,k))(state))
                    k += 1 
                if dimension==3:
                    start = k
                    end = start+shape[0]
                    inputs.append(keras.layers.Lambda(get_slice_3d(start,end))(state))
                    k = end
                    
        # 2. Compute the trend
        """ From PKF-Burgers code
        trend_u, trend_V, trend_nu = self._trend_model([u,V,nu_u_xx])
        """
        trends = self._trend_model(inputs)
        
        # 3. Outputs the trend as a single array

        # 3.1 Reshape trends
        """ from PKF-Burgers code
        trend_u = keras.layers.Reshape((1,self.input_shape_x,1))(trend_u)
        trend_V = keras.layers.Reshape((1,self.input_shape_x,1))(trend_V)
        trend_nu = keras.layers.Reshape((1,self.input_shape_x,1))(trend_nu)
        """
        
        reshape_trends = [] 
        for trend, dimension in zip(trends, dimensions):
            
            #shape = tuple(dim.value for dim in trend.shape[1:])
            # update from keras -> tensorflow.keras
            shape = tuple(dim for dim in trend.shape[1:])
            
            if dimension==1 or dimension==2:
                # for 1D fields like (128,1) transform into (1,128,1)
                # for 2D fields like (128,128,1) transform into (1,128,128,1)
                shape = (1,)+shape
            elif dimension==3:
                # 3D fields can be compated: two fields (36,128,128,1) become the single field (72,128,128,1)
                pass
            else:
                raise NotImplementedError
                
            reshape_trends.append(keras.layers.Reshape(shape)(trend))
        
        # 3.2 Concatenates all trends
        """ From PKF-Burgers code:
        trends = keras.layers.Concatenate(axis=1)([trend_u,trend_V,trend_nu])
        """
        trends = keras.layers.Concatenate(axis=1)(reshape_trends)
        
        # 2.5 Compute the model       
        self._dynamical_trend = keras.models.Model(inputs=state,outputs=trends)    
'''