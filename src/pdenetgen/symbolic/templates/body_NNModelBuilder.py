template = '''
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
        {% for coord in coordinates %}self.{{coord.input_shape}} = shape[{{coord.index}}]
        {% endfor %}
                
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

        {% if constant_functions_flag %}
        #-----------------------
        # Set constant functions
        #-----------------------
          
        # Set a default nan value for constants
        {% for function in constant_functions %}self.{{function.as_keyvar}} = np.nan # @@ set constant value @@
        {% endfor %}
                
        # Set constant function values from external **kwargs (when provided)
        for key in kwargs:
            if key in self.constant_functions:
                setattr(self, key, kwargs[key])
        
        # Alert when a constant is np.nan
        for function in self.constant_functions:
            if getattr(self, function) is np.nan:
                print(f"Warning: function `{function}` has to be set")
        {% endif %}

        {% if constants_flag %}
        #---------------------------
        # Set constants of the model
        #---------------------------
          
        # Set a default nan value for constants
        {% for constant in constants %}self.{{constant.as_keyvar}} = np.nan # @@ set constant value @@
        {% endfor %}
                
        # Set constant values from external **kwargs (when provided)
        for key in kwargs:
            if key in self.constants:
                setattr(self, key, kwargs[key])
        
        # Alert when a constant is np.nan
        for constant in self.constants:
            if getattr(self, constant) is np.nan:
                print(f"Warning: constant `{constant}` has to be set")
        {% endif %}
        
        # Set NN models
        self._trend_model = None
        self._exogenous_model = None
    
    def index(self, coord, step:int):
        """ Return int array of shift index associated with coordinate `coord` for shift `step` """
        # In this implementation, indexes are memory saved in a dictionary, feed at runtime 
        if (coord,step) not in self._index:
            self._index[(coord,step)] = (self._index[(coord,0)]+step)%self.shape[self.coordinates.index(coord)]
        return self._index[(coord,step)] 
    
'''
