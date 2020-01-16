# Author: O. Pannekoucke
# Creation: 20/03/2017

import numpy as np

import multiprocessing

class Parallel(object):
    """
    Tool for // computation
    """
    def __init__(self, f,nb_pool=8):
        self.f = f
        self.nb_pool = nb_pool

    def run(self,args):
        with multiprocessing.Pool(self.nb_pool) as pool:
            out = pool.map(self.f, args)
        return out

class Model(object):
    ''' Template for numerical model prediction of evolution dynamics

    Note
    ====
        
        Single-time-step / multiple-time-step schemes can be considered here:
        
        It is possible to consider **coupled time step** (leapfrog) as 
        well as **decoupled time step** (euler, rk4,..)
    
    '''

    '''
    .. todo :: 
            (1) Construire un dictionnaire pour les schémas. 
    '''

    def __init__(self, time_scheme='rk4'):
        self.time_schemes = { # Coupled time methods
                        'leapfrog':self._leapfrog,
                         # decoupled time methods
                        'rk4':self._rk4,
                        'rk2':self._rk2,
                        'euler':self._euler,
                        'splitting':self._time_splitting}

        self.set_time_scheme(time_scheme)

    @property
    def dt(self):
        return self._dt

    def set_dt(self, dt):
        self._dt = dt

    def set_time_scheme(self, time_scheme:str):
        if time_scheme in self.time_schemes:
            self.time_scheme =  self.time_schemes[time_scheme]
        else:
            raise ValueError(f"Time scheme {time_scheme} is not the valid time schemes: {self.time_schemes.keys()}")

    
    def trend(self, t, state):
        """
        Trend of the dynamics called during the time integration
        :param t:
        :param state:
        :return:

        """
        raise NotImplementedError()

    @staticmethod
    def _euler(trend, t, state, dt):
        """
        Euler scheme

        :param t:
        :param dt:
        :param state:
        :return: updated state
        """
        return state + dt * trend(t, state)

    @staticmethod
    def _rk2(trend, t, state, dt):
        """
        Second order Ruge Kuta scheme

        :param t:
        :param dt:
        :param state:
        :return: updated state
        """
        state_demi = state + trend(t, state) * (dt*0.5)
        return state + dt * trend(t+0.5*dt, state_demi)

    @staticmethod
    def _rk4(trend, t, state, dt):
        """
        Fourth order Runge-Kuta scheme

        :param t:
        :param dt:
        :param state:
        :return: updated state
        """
        k1 = trend(t, state)
        k2 = trend(t+dt*0.5, state+dt*0.5*k1)
        k3 = trend(t+dt*0.5, state+dt*0.5*k2)
        k4 = trend(t+dt, state+dt*k3)
        return state + (k1+k2*2+k3*2+k4)*(dt/6)

    @staticmethod
    def _leapfrog(trend, t, states, dt):
        ''' Second Order leapfrog time scheme 
        Parameter
        =========
        (u_0, u1) : the two initial state

        '''
        """
        .. todo:
            Implement a Robert-Asselin filter to limit the decoupling. 
        """
        state_0, state_1 = states
        state_2 = state_0 + trend(t, state_1) * (2*dt)
        """
        .. todo: 
            Add an Asselin filter or higher order (P. Willams)
        """
        return [state_1,state_2]

    @staticmethod
    def _time_splitting(trend, time, state, dt):
        '''
        Compute the time integration as a splitting method.

        tendance :: is a list of dictionnary like
                    {'trend':trend, 'scheme':scheme} which is a classical integration
                    or
                    {'update':update} which relies on theoretical integration over dt
        time :: float corresponding to the integration time
        x :: state
        dt :: float corresponding to the integration time step.
        '''
        for fraction_step in trend:
            if 'update' in fraction_step:
                # Case of full integration over dt
                state = fraction_step['update'](time,state,dt)
            else:
                # case of classical integration of a trend over dt
                loc_tendance = fraction_step['trend']
                loc_schema = fraction_step['scheme']
                state = loc_schema(loc_tendance,time, state, dt)
        return state

    
    def forecast(self, window, u0, saved_times=None):
        """ Time integrates a single state over a time window

        :param window: time window
        :param u0: initial state of the integration
        :param saved_times: optional saved times, default is the given time window
        :return: a dictionary of computed time steps and at saved times.
        """

        # Return a dictionary of computed time steps saved time steps over the time forecast_window.
        # update saved_times
        saved_times = self._check_saved_times(window, saved_times)
        traj = {}
        for time, next_time in zip(window[:-1], window[1:]):
            if time in saved_times: traj[time] = u0
            #
            # True for separated time steps.. 
            #
            dt = next_time - time
            #try:
            u1 = self.time_scheme(self.trend, time, u0, dt)
            u0 = u1
            #except:
            #    print(f"An exception occurs in Model.forecast at time {time}")
            #    return traj

        time = window[-1]
        if time in saved_times: traj[time] = u0
        return traj

    def predict(self, window, u0, saved_times=None):
        """ Time integrates a single state over a time window

        :param window: time window
        :param u0: initial state of the integration
        :param saved_times: optional saved times, default is the given time window
        :return: an array of computed time steps and at saved times.
        """

        # Return a dictionnary of computed time steps saved time steps over the time forecast_window.
        # update saved_times
        saved_times = self._check_saved_times(window, saved_times)
        traj = []
        for time, next_time in zip(window[:-1], window[1:]):
            if time in saved_times: traj.append(u0)
            #
            # True for separated time steps..
            #
            dt = next_time - time
            #try:
            u1 = self.time_scheme(self.trend, time, u0, dt)
            u0 = u1
            #except:
            #    print(f"An exception occurs in Model.forecast at time {time}")
            #    return traj

        time = window[-1]
        if time in saved_times: traj.append(u0)
        return np.array(traj)

    def _forecast(self, args):
        """ Internal forecast method for parallel computation """
        return self.forecast(*args)
    
    def ensemble_forecast(self, window, states, saved_times=None, parallel=True, nb_pool=8):
        """
        Ensemble forecasting of a list of given state at a given time.
        :param window:
        :param states: list of input state
        :param saved_times:
        :param parallel:
        :return:
        """

        # Return a dictionnary of computed time steps saved time steps over the time forecast_window.
        # update saved_times
        saved_times = self._check_saved_times(window, saved_times)
        forecasts = {time:[] for time in saved_times}

        if parallel:

            parallel = Parallel(self._forecast, nb_pool)
            tmp_forecasts = parallel.run([ [window, state, saved_times] for state in states])

            while tmp_forecasts:
                forecast = tmp_forecasts.pop(0)
                for time in forecast:
                    forecasts[time].append( forecast[time] )
        else:

            for state in states:
                forecast = self.forecast(window, state, saved_times)
                for time in forecast:
                    forecasts[time].append( forecast[time] )
        return forecasts
    
    def forecast_tl(self, window, state, perturbations, saved_times=None, h=1e-4):
        # update saved_times
        saved_times = self._check_saved_times(window, saved_times)
        # Reference trajectory
        forecast = self.forecast(window, state, saved_times)
        # Ensemble computation from perturbation, based on generator for memory saving
        perturbed_states = ( (state + perturbation * h) for perturbation in perturbations)
        tl_forecasts = self.ensemble_forecast(window, perturbed_states, saved_times)
        # Computation of the perturbation a time
        for q in saved_times:
            tl_forecasts[q] = [ (state - forecast[q] )/h for state in tl_forecasts[q] ]
        return forecast, tl_forecasts
    
    def randstate(self):
        '''
        Generate a model state from random initial condition
        ----------------------------------------------------
        as defined by the number of iteration for initialization _nstart
        
        '''
        # Génération d'un état aléatoire
        xo=np.random.normal(size=self.n)
        # Initialisation du temps en fonction du nombre d'itération de chauffe
        t=np.arange(self.nstart)*self.dt
        t_end = t[-1]
        return self.forecast(t, xo, saved_times=[t_end])[t_end]
    
    def window(self, end, start=0.):
        return np.arange(start, end+self.dt/3, self.dt)
    
    @staticmethod
    def _check_saved_times(window, saved_times):
        if saved_times is None:
            saved_times = window
        elif type(saved_times)==float or type(saved_times)==np.float64:
            saved_times = [saved_times]
        return saved_times

class IdentityModel(Model):
    """
        Identity model is a stationnary model which return the same initial state for any time.
    """
    _dt=1.
    def trend(self,t,state):
        return np.zeros(state.shape)

class LinearModel(Model):

    def make_M(self, window):
        M = np.zeros((self.n, self.n))
        ei = np.zeros(self.n)
        saved_times = [window[-1]]

        new = True
        if new:
            def dirac(i):
                out = np.zeros(self.n)
                out[i] = 1.
                return out
            M = self.ensemble_forecast(window, [dirac(i) for i in range(self.n)], \
                                       saved_times)[window[-1]]
            M = np.asarray(M)
        else:
            for i in range(self.n):
                ei[i] = 1.
                M[:,i] = self.forecast(window, ei, saved_times)[window[-1]]
                ei[i] = 0.

        return M

