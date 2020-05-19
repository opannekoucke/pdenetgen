"""
Author: O. Pannekoucke
Description: 
    Defined time schemes as resnet using keras.
"""

import tensorflow.keras as keras


def make_nn_rk4(dt, trend):
    """ Implementation of an RK4 with Keras """    
    
    state = keras.layers.Input(shape = trend.input_shape[1:])
    
    # k1 
    k1 = trend(state)
    # k2 
    _tmp_1 = keras.layers.Lambda(lambda x : 0.5*dt*x)(k1)
    input_k2 = keras.layers.add([state,_tmp_1])
    k2 = trend(input_k2)
    # k3 
    _tmp_2 = keras.layers.Lambda(lambda x : 0.5*dt*x)(k2)
    input_k3 = keras.layers.add([state,_tmp_2])
    k3 = trend(input_k3)
    # k4 
    _tmp_3 = keras.layers.Lambda(lambda x : dt*x)(k3)
    input_k4 = keras.layers.add([state,_tmp_3])
    k4 = trend(input_k4)
    
    # output
    # k2+k3
    add_k2_k3 = keras.layers.add([k2,k3])
    add_k2_k3_mul2 = keras.layers.Lambda(lambda x:2.*x)(add_k2_k3)
    # Add k1,k4
    _sum = keras.layers.add([k1,add_k2_k3_mul2,k4])
    # *dt
    _sc_mul = keras.layers.Lambda(lambda x:dt/6.*x)(_sum)
    output = keras.layers.add([state, _sc_mul])
    
    time_scheme = keras.models.Model(inputs =[state], 
                                     outputs=[output])
    return time_scheme 