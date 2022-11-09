# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:50:08 2022

@author: vcant

Reactive model for simple oxidation of organic carbon
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt


def reactive_model(
        c_ch2o: float,
        c_o2: float,
        c_co2: float,
        X: float,
        mu_max: float,
        K_mch2o: float,
        K_mo2: float,
        X_max: float,
        kd: float,
        Y: float,
        Yo: float,
        t_span: tuple,
        t_eval: np.ndarray,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        ):
    '''
    Reactive model for the oxidation of generic organic matter in groundwater (absence of light)
    It solves the double Monod equation for one electron acceptor and one donor with Euler integration steps,
    explicitly modelling bacterial growth and decay.
    The number of steps is adjustable by the n_sub (number of sub-steps) parameter

    Parameters
    ----------
    c_ch2o: float
        Initial CH2O concentration
    c_o2: float
        Initial O2 concentration
    c_co2: float
        initial co2 concentration
    X: float
        initial biomass concentration
    
    mu_max: float
        Maximum bacteria growth rate, according to the Monod-kinetics model
    K_mch2o: float
        K_m for CH2O (electron donor)
    K_mo2: float
        K_m of O2 (electron acceptor)
    X_max: float 
        maximum biomass carrying capacity [mg_bio/Lb]
    kd: float
        the linear bacterial decay rate
    Y: float
        biomass efficiency ratio [mg_bio/mmol_electron_donor]
    Yo: float
        the biomass efficiency ratio for the oxygen
    t_span: tuple (min and max) values for evaluation
    t_evap: array
        time intervals for evaluation of the integral
    rtol,atol: floats, tolerance parameters for ODE solver (scipy.solve_ivp)
    
    Returns
    -------
     solve_ivp solution object (Improve)
    '''
    y0 = np.array([c_ch2o,c_o2,c_co2,X])

    
    # Defining ODE function:
    def y(t, y):
        c_ch2o = y[0]
        c_o2 = y[1]
        
        mu = mu_max*(c_ch2o/(c_ch2o+K_mch2o))*(c_o2/(c_o2+K_mo2))
        # Reaction rates
        r_g = mu*X*(1-X/X_max)
        r_d = -kd*X
        
        # bacterial growth rate
        r_X = r_g + r_d
        
        # substrate and products growth rate:
        r_s = r_X/Y
        r_O = r_X/Yo
        
        return np.array([-r_s,-r_O,r_O,r_X])
    
    #solve IVP solver
    sol = scipy.integrate.solve_ivp(y, t_span, y0,method = 'RK45',
                t_eval=t_eval, rtol = rtol, atol = atol)

    return sol

if __name__ == '__main__':

    #Initial values:
        
    C_CH2O = 0.6 #mM
    C_O2 = 0.27 #mM
    C_CO2 = 0 #mM 
    'Observation: More accurate value for CO2 should be the atmosphere equilibrium [CO2]'
    
    
    # Declaring lists to store the time-step concentrations:
    ch20_l = [C_CH2O] #CH_2O concentrations
    co2_l = [C_O2] #O_2 concentrations
    cco2_l = [C_CO2] #CO_2 concentrations
    
    # time-step of simulation:
    dt = 1 #hr
    
    # rate model: Monod-kinetics:
    r_max = 1.2 #mM/hr
    K_med = 2 #mM
    K_mea = 1 #mM
    
    ## time-step arrays:
    time = np.arange(0,30+dt,dt)
    
    #setting initial values for the loop:
    c_ch2o,c_o2,c_co2 = (C_CH2O, C_O2, C_CO2)
    
    X = 1
    X_max = 5
    kd = 0.1/(30*24)
    Y = 17
    Yo = 68.493
    #Numerical calculation with solve_ivp:
        
    sol = reactive_model(C_CH2O,C_O2,C_CO2,X,r_max,K_med, K_mea,X_max,kd,Y,Yo,(time[0],time[-1]),time)
    
    ## plotting:
    fig, ax = plt.subplots(figsize = (8,6))
    
    ax.plot(time,sol.y[0,:], '--k', lw = 1.2, label = '$CH_2O$')
    ax.plot(time,sol.y[1,:], '--r', lw = 1.1, label = '$O_2$')
    ax.plot(time,sol.y[2,:], '-b', lw = 1.5, label = '$CO_2$')
    ax.set_xlim([0,30])
    ax.set_ylim([0,0.6])


