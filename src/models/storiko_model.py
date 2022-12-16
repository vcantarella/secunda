# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 13:52:09 2022

@author: Vitor Cantarella

Model for the Störiko 2021 and Störiko 2022 models of denitrification of bacterias.
"""

## Packages:
import numpy as np
import scipy


def column_model_storiko(stress_periods, parameters, time_intervals, **kwargs):
    '''
    

    Parameters
    ----------
    stress_periods : list[tchange: ndarray, c_in: ndarray, v_in: ndarray]
        tchange (n stress periods): time where a new stress period starts
        c_in (len(tchange) x nelements): concentration input at each stress period
        v_in (len(t_change) x nelements): velocity input at each stress period
    parameters : tuple
        Contains all the necessary parameters for the model to run
        except the boundary condition parameters which are defined by the stress period
    time_intervals: np.ndarray
        array with the time values where the function will be evaluated. stress period times must be inside this array
    **kwargs: arguments for the ODEsolver: scipy.integrate.odeint
    Yields
    ------
    Results
        Return the concentration vector of the outflow at the time_intervals

    '''
    
    def r_nox_monod(nu_nox,K_nox,K_doc,I_o2,C_nox,C_doc,C_o2,B):
        '''
        monod type rate for denitrification. Valid for NO3 and NO2 denitrification rates See Störiko 2021, and 2022.
        '''
        
        r = nu_nox*B*np.exp(np.log(C_nox)-np.log(C_nox+K_nox))*np.exp(np.log(C_doc)-np.log(C_doc+K_doc))*np.exp(np.log(I_o2)-np.log(I_o2+C_o2))
        return r
    
    def r_o2_monod(nu_o2,K_o2,K_doc,C_o2,C_doc,B):
        '''
        -------
        aerobic respiration rate for O2 with consumption of organic matter.

        '''
        
        r = nu_o2*B*np.exp(np.log(C_o2)-np.log(C_o2+K_o2))*np.exp(np.log(C_doc)-np.log(C_doc+K_doc))
        
        return r
    
    def r_doc_monod(r_no3,r_no2,r_o2, K_rel, C_sat, C_doc):
        r_doc = (2/7)*r_o2 + (3/14)*r_no2 + (1/7)*r_no3
        
        r_rel = K_rel*(C_sat - C_doc)
        
        return r_rel-r_doc
    
    def r_bacteria_monod(B, r_no3,r_no2,r_o2, Y_max, K_dec, B_max):
        '''
        rate calculation for bacterial cell density change.
        
        Parameters
        ----------
        B : float or nd.array
            bacterial cell density can be single value or array of cells.
        r_no3 : float
            nitrate rate.
        r_no2 : TYPE
            nitrite rate
        r_o2 : TYPE
            o2 oxidation rate.
        Y_max : array [Y_max-NO3,Y_max-NO2,Y_max-O2]
            maximum yield per reaction rate (defined in an array referencing the reactions of NO3, NO2 and O2 respectively).
        K_dec : float
            bacterial decay rate [cell/time].
        B_max: float
            maximum bacterial cell density capacity in system

        Returns
        -------
        rate: float or array, depending on B

        '''
        if isinstance(B, float): B = np.array([B])
        
        #Calculating Yield, formula: Y = Y_max*(1 - B/B_max)
        Y = np.repeat(Y_max[np.newaxis],B.shape[0], axis = 0)*(1-B[:,np.newaxis]/B_max)
        #Computing rates for growth and decay:
        r = (2/7)*r_o2*Y[:,0] + (3/14)*r_no2*Y[:,1] + (1/7)*r_no3*Y[:,2]
        r = r - K_dec*B
        
        return r
    
    def monod_derivatives(t,y,deltax,
                    D,
                    alpha_l,
                    v,
                    c_in,
                    no3_args,
                    no2_args,
                    o2_args,
                    doc_args,
                    B_args,
                    shape
                    ):
        '''
        Compute the the rates of concentration change at each cell for each compound.

        Parameters
        ----------
        t : unused (to be replaced with time)
            DESCRIPTION.
        y : nd.array (ncells x nelements)- FLATTENED
            concentration values at each cell
        deltax : float (cell-size)
            cellsize.
        D : nd.array (nelements)
            effective diffusivity.
        alpha_l : (1/L)
            dispersion coefficient.
        v : float
            velocity.
        c_in : nd.array (nelements)
            DESCRIPTION.

        Returns
        -------
        r : nd.array (y.shape)
            rate_equations at each cell for each element

        '''
        
        ##Defining transport terms:
            
        #Mixing factor (dispersion and diffusion coefficient)
        mixf = (D + v*alpha_l)/deltax**2
    
        #Initiating rates:
        r = np.zeros(shape)
        y = np.reshape(y, shape)
        
        
        #Boundary condition: Inflow (advection)+ nodiffusion boundary (dispersion):
        r[0,:-1] = -v*(y[0,:-1]-c_in)/(deltax/2) + (-mixf)*y[0,:-1] + mixf*y[1,:-1]
        
        #Finite differences dispersion for the middle portion:
        sliding_window = np.lib.stride_tricks.sliding_window_view(y[:,:-1], (3,1))
        sliding_window = sliding_window.squeeze()
        kernel = np.array([mixf, -2*mixf, mixf])
        kernel = kernel.T
        
        r[1:-1,:-1] = np.sum(sliding_window*kernel,axis = 2)
        
        #Update the last cell (No diffusion boundary):
        r[-1,:-1] = mixf*y[-2-1,:-1] + (-mixf)*y[-2,:-1]
        
        #Advection finite differences formulation
        r[1:-1,:-1] = r[1:-1,:-1]-v*(np.roll(y,-1,axis=0)[1:-1,:-1]-np.roll(y,1,axis=0)[1:-1,:-1])/(2*deltax)
        
        #Advective boundary: last term is the outflow concentration:
        r[-1,:-1] = r[-1,:-1]-v*(y[-1,:-1]-y[-2,:-1])/deltax
        
        #Defining no change from transport for the Bacteria:
        r[:,-1] = 0
        
        
        
        ## Defining the reaction terms:
        #Unpacking Parameters
        nu_no3,K_no3,K_doc_3,I_o2_3 = no3_args
        
        nu_no2,K_no2,K_doc_2,I_o2_2 = no2_args
        
        nu_o2,K_o2_o,K_doc_o = o2_args
        
        K_rel,C_sat = doc_args
        
        Y_max, K_dec, B_max = B_args
        
        #Defining reaction rates:
        ##Nitrate
        r_no3 = r_nox_monod(nu_no3,K_no3,K_doc_3,I_o2_3, y[:,0],y[:,3],y[:,2],y[:,4])
        ##NO2
        r_no2 = r_nox_monod(nu_no2,K_no2,K_doc_2,I_o2_2, y[:,1],y[:,3],y[:,2],y[:,4])
        ##O2
        r_o2 = r_o2_monod(nu_o2,K_o2_o,K_doc_o,y[:,2],y[:,3],y[:,4])
        ##DOC
        r_doc = r_doc_monod(r_no3,r_no2,r_o2, K_rel, C_sat, y[:,3])
        
        ##dBdt
        dBdt = r_bacteria_monod(y[:,4], r_no3,r_no2,r_o2, Y_max, K_dec, B_max)
        dBdt = 0
        
        #r_no3, r_no2, r_o2, r_doc, dBdt  = np.array([0,0,0,0,0], dtype=np.float32)
        
        r[:,0] = r[:,0] - r_no3
        r[:,1] = r[:,1] + r_no3 - r_no2
        r[:,2] = r[:,2] - r_o2
        r[:,3] = r[:,3] + r_doc
        r[:,4] = r[:,4] + dBdt
        
        return r.flatten()
    
    def calculate_stress_period(time_intervals,y0, f, f_args,
                                **kwargs):
        
        solution = scipy.integrate.odeint(f,y0,t = time_intervals,args = f_args, tfirst = True,
                              **kwargs)
        #We just want the last term which is the outflow, but we need the rest for the next stress period
        return solution
    
    #Unpacking stress_periods
    tchange = stress_periods[0]
    c_in_array = stress_periods[1]
    v_in = stress_periods[2]
    
    #Unpacking parameters
    length = parameters['length']
    ncells = parameters['ncells']
    D = parameters['D']
    alpha_l = parameters['alpha_l']
    no3_args = parameters['no3_args']
    no2_args = parameters['no2_args']
    o2_args = parameters['o2_args']
    doc_args = parameters['doc_args']
    B_args = parameters['B_args']
    
    #Defining the coordinate system and initial values
    deltax = length/ncells
    #xcoords = np.arange(deltax/2,length, deltax)
    B_ini = 3.3e9
    y0 = y0 = np.repeat(1e-11,(ncells)*5).reshape((ncells,5))
    y0[:,4] = np.repeat(B_ini, ncells)
    
    shape = y0.shape
    
    y0 = y0.flatten()
    
    solutions = np.zeros((len(time_intervals),5))
    
    #Calculating each stress period
    for i in range(len(tchange)):
        #stress periods
        if i < len(tchange)-1:
            t_intervals = time_intervals[np.where((time_intervals >= tchange[i]) & (time_intervals <= tchange[i+1]))]
        else:
            t_intervals = time_intervals[np.where((time_intervals >= tchange[i]))]
        
        stress_intervals = t_intervals.copy() - t_intervals[0]
        
        v = v_in[i]
        c_in = c_in_array[i,:]
        f_args = (deltax,
                  D, alpha_l,
                  v,
                  c_in,
                  no3_args,
                  no2_args,
                  o2_args,
                  doc_args,
                  B_args,
                  shape)
                    
        sol = calculate_stress_period(stress_intervals,y0 = y0, f= monod_derivatives,
                                f_args  = f_args, **kwargs)
        y0 = sol[-1,:]
        
        sol = sol.reshape((sol.shape[0],shape[0],shape[1]))
 
        solutions[np.in1d(time_intervals,t_intervals),:] = sol[:,-1,:]
    
    return solutions
        

if '__int__' == '__main__':
    
    pass
    
    
    
    
    
    
    
    
    
