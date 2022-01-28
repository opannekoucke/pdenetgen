template = '''
    def _make_trend_model(self):
        """ Generate the NN used to compute the trend of the dynamics """
        {% if constant_functions_flag -%} 
        # Alias for constant functions
        #-----------------------------
        {% for function in constant_functions -%}
        {{function.as_keyvar}} = self.{{function.as_keyvar}}
        if {{function.as_keyvar}} is np.nan:
            raise ValueError("Constant function '{{function.as_keyvar}}' is not set")
        {% endfor %}          
        
        {% endif %}                
        {% if constants_flag -%}
        # Alias for constants
        #--------------------
        {% for constant in constants -%}
        {{constant.as_keyvar}} = self.{{constant.as_keyvar}}
        if {{constant.as_keyvar}} is np.nan:
            raise ValueError("Constant '{{constant.as_keyvar}}' is not set")
        {% endfor %}         
        {% endif %}        
        
        # Set input layers
        #------------------
        # Set Alias for coordinate input shapes
        {% for coord in coordinates -%}
        {{coord.input_shape}} = self.{{coord.input_shape}}
        {% endfor %}        
        # Set input shape for prognostic functions
        {% for function in prognostic_functions -%}
        {{function.as_keyvar}} = keras.layers.Input(shape =({% for shape in function.input_shape %}{{shape}},{% endfor %}))
        {% endfor %} 
        {% if constant_functions_flag -%}
        # Set input shape for constant functions 
        {% for function in constant_functions -%}
        {{function.as_keyvar}} = keras.layers.Input(shape =({% for shape in function.input_shape %}{{shape}},{% endfor %}))
        {% endfor %}{% endif %}        
        {% if exogenous_functions_flag -%}
        # Set input shape for exogenous functions 
        {% for function in exogenous_functions -%}
        {{function.as_keyvar}} = keras.layers.Input(shape =({% for shape in function.input_shape %}{{shape}},{% endfor %}))
        {% endfor %}{% endif %}
         
        # Keras code
         
        # 2) Implementation of derivative as ConvNet
        # Compute derivative
        #-----------------------
        #
        #  Warning: might be modified to fit appropriate boundary conditions. 
        #
        {% for derivative in spatial_derivatives -%}
        {% for line in derivative.as_code %}{{line}}
        {% endfor %}
        {% endfor %}        
        
        # 3) Implementation of the trend as NNet
        
        {% for line in trend_code -%}
        {{line}}
        {% endfor%}
        
        # 4) Set 'input' of model
        inputs = [
                # Prognostic functions
                {% for function in prognostic_functions -%}     
                {{function.as_keyvar}},{% endfor %}
                {% if exogenous_functions_flag -%}
                # Exogenous functions 
                {% for function in exogenous_functions -%}    
                {{function.as_keyvar}},{% endfor %}
                {% endif -%}{% if constant_functions_flag -%}
                # Constant functions
                {% for function in constant_functions -%}    
                {{function.as_keyvar}},{% endfor %}{% endif %}
            ]         
   
        # 5) Set 'outputs' of model 
        outputs = [
            {% for function in prognostic_functions -%}
            {{function.trend}},
            {% endfor -%}
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
        {% for function in prognostic_functions -%}
        d{{function.as_keyvar}} = dstate[{{function.index}}]
        {% endfor %}

        # Load physical functions from state
        #------------------------------------
        {% for function in prognostic_functions -%}
        {{function.as_keyvar}} = state[{{function.index}}]
        {% endfor %}          
         
        {% if exogenous_functions_flag %}
        # Compute exogenous functions
        #-----------------------------
        exogenous = self.compute_exogenous(t, state) # None if no exogenous function exists
        {% for function in exogenous_functions -%}
        {{function.as_keyvar}} = exogenous[{{function.index}}]
        {% endfor %}{% endif %}
        
        # Compute the trend value from model.predict
        #-------------------------------------------  
        inputs = [
            # Prognostic functions
                {% for function in prognostic_functions -%}{{function.as_keyvar}},
                {% endfor -%}{% if exogenous_functions_flag -%}
            # Exogenous functions 
                {% for function in exogenous_functions %}{{function.as_keyvar}},
                {% endfor %}{% endif -%}{% if constant_functions_flag -%}
            # Constant functions 
                {% for function in constant_functions %}self.{{function.as_keyvar}},
                {% endfor %}{% endif -%}
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
                
    {% if exogenous_functions_flag %}
    
    def _make_exogenous_model(self):
        raise NotImplementedError
    
    def compute_exogenous(self, t, state):
        """ Computation of exogenous functions """
        if self._exogenous_model is None:
            self._make_exogenous_model()
        
        raise NotImplementedError # if no exogenous functions are needed
    {% endif %}

'''

# todo: Updates name of method keras model for trend.
''' 
  * [ ] change names of keras models for trend so to facilitate their use
        - _trend_model -> _trend_from_list ?
        - _dynamical_trend -> _trend_from_array ? 
'''
