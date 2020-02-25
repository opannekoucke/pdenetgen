#!/usr/bin/env python
# coding: utf-8

# <h1><center> Training of a closure for the uncertainty ropagation in the Burgers equation</center></h1>
# <center>
#     Olivier Pannekoucke <br>
# </center>

# <div class="math abstract">
#     <p style="text-align:center"><b>Abstract</b></p>
#     <p>
#     Estimation of the closure for the uncertainty propagation in the Burgers dynamics
#     </p>
# </div>

# ---
# <center> <b>Table of contents</b> </center>
# 
#  1. [Introduction](#introduction)
#  1. [The Burgers dynamics and its unclosed PKF formulation](#burgers-pkf)
#   1. [Set of the function and symbols](#burgers-pkf-sympy-definition)
#   1. [Set constants for numerical experiments](#burgers-pkf-num-definition)
#   1. [Set of the Burgers equation](#burgers-pkf-dyn-burgers)
#   1. [Set of the PKF equations for the Burgers equation](#burgers-pkf-dyn-PKF)
#  1. [Application numerique](#num)
#   1. [Définition et utilisation de la dynamique fermée](#num-closed)
#   1. [Définition et apprentissage de la dynamique non-fermée](#num-unclosed)
# ---

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#%matplotlib notebook


# ## Introduction <a id='introduction'>

# The aim is to design a NN which merges known and unknown physics.

# In[2]:


import sympy
from sympy import (Function, symbols, init_printing, Derivative, 
                   latex, Add, Mul, Pow, 
                   Integer, Rational, Float, Symbol, symbol,
                   srepr, Tuple
                  )
init_printing() 


# ### Functions for plot

# In[3]:


import matplotlib.pyplot as plt


# In[4]:


def plot_results(data, label=None, labelx=True, title=None, save_file=None, normalisation=None, 
                 selected_times=None,style=None, name=None, alpha=1., bolds=[0., 1.]):
    
    normalisation = 1. if normalisation is None else normalisation
                 
    selected_times = [time for time in data] if selected_times is None else selected_times
                 
    style = 'k' if style is None else style
                 
    for time in selected_times:
        lalpha = alpha if time in bolds else 0.2
        lname = name if time==selected_times[-1] else None
        plt.plot(domain.x[0],data[time]/normalisation, style, alpha = lalpha, label=lname)
                 
    if labelx:
        plt.xlabel('$x/D$', fontsize=15)
    if label:
        plt.ylabel(label, fontsize=15)
    if title:
        plt.title(title)
    if save_file:
        plt.savefig(save_file)


# ## The Burgers dynamics and its unclosed PKF formulation <a id='burgers-pkf'/>

# In[5]:


from pdenetgen import NNModelBuilder, Eq


# In[6]:


def display_system(system):
    print(50*'*')
    for equation in system:
        display(equation)
        print(50*'*')


# #### Set of the function and symbols <a id='burgers-pkf-sympy-definition'>

# In[7]:


t, x = symbols('t x')

u = Function('u')(t,x)
closure = sympy.Function('closure')(t,x)
V = Function('{V_{u}}')(t,x)
nu = Function('{\\nu_{u,xx}}')(t,x)
Kappa = symbols('\\kappa')


# #### Set constants for numerical experiments <a id='burgers-pkf-num-definition'>

# In[8]:


# Constant setting following Pannekoucke et al. (2018)
n = 241
kappa = 0.0025
dt = 0.002


# #### Set of the Burgers equation <a id='burgers-pkf-dyn-burgers'>

# In[9]:


burgers_dynamics = [
        Eq(
        Derivative(u,t),
        Kappa*Derivative(u,x,2)-u*Derivative(u,x)
      ),
]
display_system(burgers_dynamics)


# In[ ]:


burgers_NN_builder = NNModelBuilder(burgers_dynamics, "Burgers")
print(burgers_NN_builder.code)
exec(burgers_NN_builder.code)
burgers = Burgers(shape=(n,), kappa=kappa)


# ##### Example of forecast from a given initial condition

# In[ ]:


domain = burgers
# Set initial condition for 'u'
U0=0.25*( 1+np.cos(2*np.pi/ domain.lengths[0]  *(domain.x[0]-0.25)) )
Umax = U0.max()


# In[ ]:


burgers.set_dt(dt)
end_time_forecast = 1.
times = burgers.window(end_time_forecast)
saved_times = times[::50]
print('saved_times :' ,saved_times)


# In[ ]:


forecast = burgers.forecast(times, np.array([U0.reshape((1,)+U0.shape+(1,)) ]))


# In[ ]:


for time in times:
    plt.plot(domain.x[0], forecast[time][0,0,:,0])


# #### Set of the PKF equations for the Burgers equation <a id='burgers-pkf-dyn-pkf'>

# In[ ]:


# From Pannekoucke et al. (2018)

pkf_dynamics = [
    # Trend of the expectation of 'u'
    Eq(
        Derivative(u,t),
        Kappa*Derivative(u,x,2)-u*Derivative(u,x)-Derivative(V,x)/Integer(2)
      ),
    # Trend of the variance
    Eq(
        Derivative(V,t),
        -Kappa*V/nu + Kappa*Derivative(V,x,2)-Kappa*Derivative(V,x)**Integer(2)/(Integer(2)*V)
        -u*Derivative(V,x)-Integer(2)*V*Derivative(u,x)
      ),
    # Trend of the diffusion
    Eq(
        Derivative(nu,t),
        Integer(4)*Kappa*nu**Integer(2)*closure
        -Integer(3)*Kappa*Derivative(nu,x,2)
        -Kappa
        +Integer(6)*Kappa*Derivative(nu,x)**Integer(2)/nu
        -Integer(2)*Kappa*nu*Derivative(V,x,2)/V
        +Kappa*Derivative(V,x)*Derivative(nu,x)/V
        +Integer(2)*Kappa*nu*Derivative(V,x)**Integer(2)/V**Integer(2)
        -u*Derivative(nu,x)
        +Integer(2)*nu*Derivative(u,x)
    )
]

display_system(pkf_dynamics)


# In[ ]:


pkf_NN_builder = NNModelBuilder(pkf_dynamics,'NN_Unclosed_PKF_Burgers')


# In[ ]:


print(pkf_NN_builder.code)
exec(pkf_NN_builder.code)


# #### Construction of a closure as a NN from parameterized form

# <div class='math philo'>
#   <p>
#   L'objectif est de calculer le terme
#   $$a\frac{\frac{\partial^{2}}{\partial x^{2}} \operatorname{{\nu_{u,xx}}}{\left(t,x \right)}}{\operatorname{{\nu_{u,xx}}}^{2}{\left(t,x \right)}} +b \frac{1}{ \operatorname{{\nu_{u,xx}}}^{2}{\left(t,x \right)}} +c\frac{ \left(\frac{\partial}{\partial x} \operatorname{{\nu_{u,xx}}}{\left(t,x \right)}\right)^{2}}{\operatorname{{\nu_{u,xx}}}^{3}{\left(t,x \right)}},$$
#   sous la forme d'une fonction exogène.
#   </p>
# </div>

# In[ ]:


class ClosedPKFBurgers(NN_Unclosed_PKF_Burgers):
    def _make_exogenous_model(self):
                
        u = keras.layers.Input(shape=(self.input_shape_x,1))
        V = keras.layers.Input(shape=(self.input_shape_x,1))
        nu_u_xx = keras.layers.Input(shape=(self.input_shape_x,1))        
        
        #
        # Computation of the spatial derivatives
        #
        
        kernel_Dnu_u_xx_x_o2 = np.asarray([self.dx[self.coordinates.index('x')]**(-2),
        -2/self.dx[self.coordinates.index('x')]**2,
        self.dx[self.coordinates.index('x')]**(-2)]).reshape((3,)+(1,1))
        Dnu_u_xx_x_o2 = DerivativeFactory((3,),kernel=kernel_Dnu_u_xx_x_o2,name='Dnu_u_xx_x_o2')(nu_u_xx)
        
        kernel_Dnu_u_xx_x_o1 = np.asarray([-1/(2*self.dx[self.coordinates.index('x')]),0.0,
        1/(2*self.dx[self.coordinates.index('x')])]).reshape((3,)+(1,1))
        Dnu_u_xx_x_o1 = DerivativeFactory((3,),kernel=kernel_Dnu_u_xx_x_o1,name='Dnu_u_xx_x_o1')(nu_u_xx)        
        
        #
        # Design of the unknown closure to train
        #
        
        # Terme 1
        div_14 = keras.layers.Lambda(lambda x: 1/x,name='DivLayer_14')(nu_u_xx)
        pow_12 = keras.layers.multiply([div_14,div_14,] ,name='PowLayer_12')
        term1 = keras.layers.multiply([pow_12,Dnu_u_xx_x_o2],name='MulLayer_25')
        
        # Terme 2
        div_13 = keras.layers.Lambda(lambda x: 1/x,name='DivLayer_13')(nu_u_xx)
        term2 = keras.layers.multiply([div_13,div_13,] ,name='PowLayer_11')
        
        # Terme 3
        pow_13 = keras.layers.multiply([Dnu_u_xx_x_o1,Dnu_u_xx_x_o1,] ,name='PowLayer_13')
        div_15 = keras.layers.Lambda(lambda x: 1/x,name='DivLayer_15')(nu_u_xx)
        pow_14 = keras.layers.multiply([div_15,div_15,div_15,] ,name='PowLayer_14')
        term3 = keras.layers.multiply([pow_13,pow_14],name='MulLayer_26')
        
        # Product by (a,b,c), implemented as Conv1D
        term1 = keras.layers.Conv1D(1,1,name='times_a',padding='same',use_bias=False,activation='linear')(term1)
        term2 = keras.layers.Conv1D(1,1,name='times_b',padding='same',use_bias=False,activation='linear')(term2)
        term3 = keras.layers.Conv1D(1,1,name='times_c',padding='same',use_bias=False,activation='linear')(term3)
                
        closure = keras.layers.add([term1, term2, term3],name='Closure')        
                
        self._exogenous_model = keras.models.Model(inputs=[u,V,nu_u_xx], outputs=[closure])
        
    def compute_exogenous(self, t, state):
        
        if self._exogenous_model is None:
            self._make_exogenous_model()
        
        u,V,nu = state
        closure = self._exogenous_model.predict([u,V,nu])
        if not isinstance(closure, list):
            closure = [closure]
        return closure
        
    def _make_full_trend(self):
        
        if self._trend_model is None:
            self._make_trend_model()
        if self._exogenous_model is None:
            self._make_exogenous_model()
        
        state = keras.layers.Input(shape=(3,self.input_shape_x,1))
        
        u = keras.layers.Lambda(lambda x : x[:,0,:,:])(state)
        V = keras.layers.Lambda(lambda x : x[:,1,:,:])(state)
        nu_u_xx = keras.layers.Lambda(lambda x : x[:,2,:,:])(state)
        
        closure = self._exogenous_model([u,V,nu_u_xx])
        trend_u, trend_V, trend_nu = self._trend_model([u,V,nu_u_xx,closure])
        
        trend_u = keras.layers.Reshape((1,self.input_shape_x,1))(trend_u)
        trend_V = keras.layers.Reshape((1,self.input_shape_x,1))(trend_V)
        trend_nu = keras.layers.Reshape((1,self.input_shape_x,1))(trend_nu)
        
        trend = keras.layers.Concatenate(axis=1)([trend_u,trend_V,trend_nu])
        self._full_trend = keras.models.Model(inputs=state,outputs=trend)


# In[ ]:


closed_burgers = ClosedPKFBurgers(shape=(241,),kappa=kappa)


# In[ ]:


closed_burgers._make_full_trend()


# **Set initial PKF fields**

# In[ ]:


# Set initial condition for the variance parameter 'V_u'
V0 = (0.01*Umax)**2 + 0*U0

# Set the initial condition for the diffusion 
# L**2 = 2nu t => nu = 0.5*L**2
lh = 0.02*domain.lengths[0]
nu0 = 0.5*lh**2 + 0*U0

state0 = np.asarray([U0, V0,nu0])
normalization = {
                'Velocity':U0.max(), 
                'Variance':V0.max(), 
                'Length-scale':lh
                }


# In[ ]:


length_scale = lambda nu: np.sqrt(2*nu)
plt.figure(figsize=(12,12))
for k,field in enumerate(normalization):
    plt.subplot(221+k)
    if field=='Length-scale':
        data = {0:length_scale(state0[k])}
    else:
        data = {0:state0[k]}
    plot_results(data, label=field)


# ## Application numérique <a id='num'/>

# In[ ]:


def plot_pkf_traj_ensemble(traj):
    plt.figure(figsize=(15,5))
    for k,field in enumerate(normalization):
        if field=='Length-scale':
            data = {time:length_scale(traj[time][k]) for time in traj}
        else:
            data = {time:traj[time][k] for time in traj}
        plt.subplot(131+k)
        plot_results(data,label=field,normalisation=normalization[field])


# ### Définition et utilisation de la dynamique fermée <a id='num-closed'/>

# In[ ]:


state0 = np.asarray([U0.reshape((1,)+U0.shape+(1,)), 
                     V0.reshape((1,)+V0.shape+(1,)), 
                     nu0.reshape((1,)+nu0.shape+(1,))])


# In[ ]:


def plot_pkf_traj_NN(traj):
    plt.figure(figsize=(15,5))
    for k,field in enumerate(normalization):
        if field=='Length-scale':
            data = {time:length_scale(traj[time][k][0,:,0]) for time in traj}
        else:
                data = {time:traj[time][k][0,:,0] for time in traj}
        plt.subplot(131+k)
        plot_results(data,label=field,normalisation=normalization[field])
    
#plt.savefig("./figures/NN-PKF-closure_loc-gaussian.jpg")    


# ## Generation of a database <a id='set-database'/>

# ##### **Gaussian random vector of Gaussian correlation function**

# In[ ]:


# Création d'une matrice de covariance d'erreur de prévision initiale: $P_0$
#   Cette matrice est construite comme une matrice homogène de corrélation Gaussienne et de longueur de portée l_h

# 1) Définition de la fonction de corrélation homogène
gauss = lambda x : np.exp(-0.5*x**2/lh**2) # lh has been previously specified 
correlation = gauss(domain.x[0]-domain.x[0][domain.shape[0]//2])
spectrum = np.abs(np.fft.fft(correlation))

# 2) Construction de B^(1/2)
std_spectrum = np.sqrt(spectrum)
def make_sample():
    zeta = np.random.normal(size=domain.shape)
    zeta = np.fft.fft(zeta)
    ef = np.fft.ifft(std_spectrum * zeta)
    ef = np.real(ef)
    return ef


# In[ ]:


plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(domain.x[0], correlation)
plt.title('Homogenous correlation function');
plt.subplot(122)
for k in range(10):
    plt.plot(domain.x[0], make_sample())
plt.title("Example of sample errors");


# ##### **Diagnosis tool for ensemble estimation of expectation/variance/diffusion tensor**

# In[ ]:


def make_init_ensemble(Ne):
    return np.array([make_sample() for k in range(Ne)])

def estimate_covariance(ensemble):
    mean = ensemble.mean(0)
    error = (ensemble - mean)/np.sqrt(len(ensemble))
    return error.T @ error

class EnsembleDiagnosis(object):
    
    def __init__(self, ensemble, base_space):
        self.base_space = base_space
        
        if isinstance(ensemble, list):
            ensemble = np.array(ensemble)
        
        if len(ensemble.shape)==3:
            ensemble = np.array([elm[0] for elm in ensemble])
        
        # 1) Computation of the mean
        self.mean = ensemble.mean(axis=0)
        
        # 2) Computation of the variance
        self.std = ensemble.std(axis=0)
        self.variance = self.std*self.std
        
        # 3) Computation of the metric terms 
        #  we use the formula g_ij = E[(D_i eps)(D_j eps)]
        
        #  a) Computation of the normalized error
        epsilon = (ensemble-self.mean)/self.std
        
        #  b) Computation of derivatives
        n = self.base_space.shape[0]
        K = np.arange(n)
        kp = (K+1)%n
        km = (K-1)%n
        dx = self.base_space.dx[0]
        Depsilon = np.array([(eps[kp]-eps[km])/(2*dx) for eps in epsilon])
        self.metric = (Depsilon*Depsilon).mean(axis=0)     # see Pannekoucke et al. (2018) for details   
        
        # Computation of the diffusion tensor
        self.diffusion = 0.5*1/self.metric
        self.length_scale = np.sqrt(2*self.diffusion)


# ##### **Ensemble validation for the covariance setting**

# In[ ]:


Ne = 1600

ensemble = make_init_ensemble(Ne)

mean = ensemble.mean(axis=0)
std = ensemble.std(axis=0)

print(f"Validation of the mean (=0): {mean.mean()} +/- {mean.std()}" )
print(f"Validation of the standard-deviation (=1): {std.mean()} +/- {std.std()}" )

ens_diagnosis = EnsembleDiagnosis(ensemble, domain)
nu_h = 0.5*lh**2

plt.figure(figsize=(15,5))

plt.subplot(131)
plt.plot(ens_diagnosis.mean)
plt.title('Moyenne')

plt.subplot(132)
plt.plot(ens_diagnosis.variance)
plt.title('Variance')

plt.subplot(133)
plt.plot(ens_diagnosis.diffusion/nu_h)
plt.title('diffusion (normalisée par $nu_h$)')


# **Computation of a large ensemble (1600 members) to build a reference**

# In[ ]:


# Standard deviation for the initial perturbation
sigma_f = 0.01*U0.max()


# In[ ]:


# Set parameters for ensemble estimation
large_Ne = 1600

# 1. Set the initial background state
random_U0 = U0 + sigma_f*make_init_ensemble(1)[0]

# 2. Build an ensemble of initial perturbed state
ensemble = make_init_ensemble(large_Ne)
ensemble_ea = np.array([random_U0+sigma_f*ea for ea in ensemble])
ensemble_ea = ensemble_ea.reshape((1,)+ensemble_ea.shape+(1,))
print(f"shape of ensemble_ea: {ensemble_ea.shape}")

# 3. Build the ensemble of forecast using the NN architecture
ensemble_forecast = burgers.forecast(times,ensemble_ea)


# In[ ]:


# 4. Compute diagnosis from ensemble
ensemble_traj = {}
for time in times[::50]:
    diagnosis = EnsembleDiagnosis(ensemble_forecast[time][0,:,:,0], domain)
    ensemble_traj[time] = [diagnosis.mean, diagnosis.variance, diagnosis.diffusion]


# In[ ]:


plot_pkf_traj_ensemble(ensemble_traj)


# #### **Generation of the training data set**

# In[ ]:


def generate_data(k, Ne=400):
    # 1. Set the initial background state
    random_U0 = U0 + sigma_f*make_init_ensemble(1)[0]

    # 2. Build an ensemble of initial perturbed state
    ensemble = make_init_ensemble(Ne)
    ensemble_ea = np.array([random_U0+sigma_f*ea for ea in ensemble])
    ensemble_ea = ensemble_ea.reshape((1,)+ensemble_ea.shape+(1,))    
        
    # 3. Compute the ensemble of forecasts
    ensemble_forecast = burgers.forecast(times,ensemble_ea)

    # 4. Compute the diagnosis    
    diagnosis_list = []
    for time in times:
        diagnosis = EnsembleDiagnosis(ensemble_forecast[time][0,:,:,0], domain)
        diagnosis_list.append( np.array([diagnosis.mean, diagnosis.variance, diagnosis.diffusion]))
        
    return diagnosis_list


# In[ ]:


data_size = 400  # for Ne=400, this takes 1h09'01'' so take care with this..
save_file = "pkf-dataset.npy"

generate_data_set = False
parallel_diagnosis = False
    
try:
    # load data
    data = np.load(save_file)
    data = data.reshape(data.shape+(1,))
except:
    # 1. Generate data   
    #data = [generate_data(k) for k in range(data_size)]
    data = []
    for k in range(data_size):
        if k%5==0:
            print(k)
        data.append(generate_data(k))
    

    # 2. Save data
    data = np.array(data)
    np.save(save_file,data)
    


# In[ ]:


data.shape


# In[ ]:


def plot_training_data():
    plt.figure(figsize=(15,5))

    title = ['Ensemble mean','Variance','Diffusion']
    normalization = [1, sigma_f**2, 0.5*lh**2]

    for trajectory in data[:10]:
        for date in  trajectory[::100]:    
            for k,field in enumerate(date):
                plt.subplot(131+k)
                plt.plot(domain.x[0], field/normalization[k] )
                plt.title(title[k])
    plt.savefig('./figures/NN-burgers-training-data.pdf')
plot_training_data()


# #### Construction d'un schéma d'intégration RK4

# In[ ]:


# Schéma temporelle de type RK4
def make_time_scheme(dt, trend):
    """ Implémentation d'un schéma de RK4 sous forme de réseau de neurones """
    import keras
    
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
    
    time_scheme = keras.models.Model(inputs =[state], outputs=[output])
    return time_scheme 


# In[ ]:


closed_pkf_burgers = ClosedPKFBurgers(shape=(241,),kappa=kappa)


# In[ ]:


closed_pkf_burgers._make_full_trend()
closed_pkf_burgers._full_trend.summary()


# In[ ]:


closed_pkf_burgers._exogenous_model.summary()


# In[ ]:


time_scheme = make_time_scheme(dt, closed_pkf_burgers._full_trend)
#time_scheme.summary()


# #### Constitution de la base de données d'apprentissage

# In[ ]:


data[0].shape


# In[ ]:


select_from = 400 # 200
X = np.array([elm[select_from:-1] for elm in data])
Y = np.array([elm[select_from+1:] for elm in data])


# In[ ]:


X = X.reshape((np.prod(X.shape[:2]),3,241,1))
Y = Y.reshape((np.prod(Y.shape[:2]),3,241,1))


# In[ ]:


X.shape


# #### **Training of the NN**

# In[ ]:


# Expérience d'apprentissage:
# 2. Adam
lr = 0.1 # old value: 0.001
epochs = 30

for iteration in range(3):
    # 1. Set the learning
    time_scheme.compile(optimizer=keras.optimizers.Adam(lr=lr),
        loss='mean_squared_error') # Ne permet pas la convergence
    
    # 2. Train
    history = time_scheme.fit(X,Y,epochs=epochs, batch_size=32,verbose=0)
    print(f"iteration {iteration} is complet")
    
    # 3. Plot history    
    plt.figure()
    plt.plot(history.history['loss'])
    
    # 4. Update the learning rate for next iteration
    lr = lr/10
    


# **Compare the weights with the previous theoretical closure**

# The weights of the theoretical closure are : 1, 3/4, -2

# In[ ]:


trained = unclosed_burgers._full_trend.get_weights()
trained = np.array((trained[0], trained[1], trained[2])).flatten()
trained


# In[ ]:


theoretical = np.array([1,3/4,-2])


# In[ ]:


relative_error = (trained - theoretical)/theoretical


# In[ ]:


relative_error*100


# **Exemple de prévision réalisée avec le modèle calibré**

# In[ ]:


# default
unclosed_burgers.set_dt(dt)
times = unclosed_burgers.window(1)
#saved_times = times[::25]
saved_times = times[::50]
print('saved_times :' ,saved_times)


# In[ ]:


trained_unclosed_traj = unclosed_burgers.forecast(times, state0, saved_times)


# In[ ]:


normalization


# In[ ]:


# PKF using trained closure
plot_pkf_traj_NN(trained_unclosed_traj)
plt.savefig('./figures/burgers-b.pdf')


# In[ ]:


# ensemble of forecast statistics
plot_pkf_traj_ensemble(ensemble_traj)
plt.savefig('./figures/burgers-a.pdf')


# ## Conclusion <a id='conclusion'/>

# <div class='math conclusion'>
#     In this notebook, a closure for the Burgers dynamics has been learned from the data. 
#     </div>    

# ## Appendix

# In[ ]:


from IPython.core.display import HTML
with open("css/lecture.css",'r') as css_file:
    css_style = css_file.read()
HTML(css_style)

