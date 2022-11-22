# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:50:08 2022

@author: vcant

Model from Kornaros 1998> Kinetic modelling of Pseudomonas Denitrificans growth and denitrification under aerobic
anoxic and transient conditions 
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

Vector = list[float]

def reactive_model(
        Sg:float,
        n1:float,
        n2: float,
        co: float,
        X: float,
        mu_aer: float,
        aerobic_pars: Vector,
        n1_pars: Vector,
        n2_pars: Vector,
        yield_pars: Vector,
        co_pars: Vector,
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
    y0 = np.array([Sg,n1,n2,co,X, mu_aer])
    alpha_max2, mu_max, Ks, Ko, Ki1, Ki2, Ki3 = aerobic_pars
    mu_m1, Kn1, Kio1 = n1_pars
    mu_m2, Kn2, Kin1, Kio2, v_max, Kio3 = n2_pars
    Y1,Y2,Ys,Ysn1,Ysn2,Yo = yield_pars
    c_star,kl = co_pars

    
    # Defining ODE function:
    def y(t, y):
        
        mu_t = mu_max*(Sg/(Ks+Sg))*(co/(co+Ko))*(Ki1/(Ki1+n1))*(Ki2/(Ki2+n2))
        if mu_aer > mu_t:
            dmu_aerdt = alpha_max2*(Ki3/(Ki3 + n2))*(mu_t - mu_aer)
        else:
            mu_aer = mu_t
            dmu_aerdt = 0
        mu_n1 = mu_m1*(n1/(n1+Kn1))*(Sg/(Ks+Sg))*(Kio1/(Kio1+co))
        mu_n2 = mu_m2*(n2/(n2+Kn2))*(Sg/(Ks+Sg))*(Kin1/(Kin1+n1))*(Kio2/(Kio2+co))
        v_n2 = v_max*(n2/(n2+Kn2))*(Sg/(Ks+Sg))*(Kio3/(Kio3+co))
        
        dXdt = (mu_aer + mu_n2 + mu_n1)*X
        dn1dt = -(1/Y1)*mu_n1*X
        dn2dt = ((1/Y1)*mu_n1-((1/Y2)*mu_n2+v_n2))*X
        dsgdt = -(1/Ys)*mu_aer*X - Ysn1/Y1*mu_n1*X \
            -Ysn2*((1/Y2)*mu_n2+v_n2)*X
        dcodt = kl*(c_star - co)-mu_aer/Yo*X
        

        return np.array([dsgdt,dn1dt,dn2dt,dcodt,dXdt,dmu_aerdt])
    
    #solve IVP solver
    sol = scipy.integrate.solve_ivp(y, t_span, y0,method = 'RK45',
                t_eval=t_eval, rtol = rtol, atol = atol)

    return sol



if __name__ == '__main__':




