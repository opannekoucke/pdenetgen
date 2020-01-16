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
        model.trainable = False
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
