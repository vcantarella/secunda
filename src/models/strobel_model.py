# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 13:52:09 2022

@author: Vitor Cantarella

Model for the Strobel 2022 models of denitrification of bacterias in flow through cell experiment.
"""

## Packages:
import jax
from jax import jit
import jax.numpy as jnp
import scipy
import numpy as np
# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)


def column_model_strobel(stress_periods, parameters, time_intervals, **kwargs):
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
    @jit
    def r_nox_doc(mu_nox,Y_lac,K_nox,K_lac,ne,rho_s,C_nox,C_doc,B):
        '''
        monod type rate for denitrification. Valid for NO3 and NO2 denitrification rates See Störiko 2021, and 2022.
        '''
        
        r = mu_nox/Y_lac*B*jnp.log(jnp.exp((C_nox)/(C_nox+K_nox)))*jnp.log(jnp.exp((C_doc)/(C_doc+K_lac)))*(1-ne)*rho_s/ne
        return r
    @jit
    def r_nox_soc(mu_nox,Y_soc,K_nox,ne,rho_s,C_nox,B):
        '''
        monod type rate for denitrification. Valid for NO3 and NO2 denitrification rates See Störiko 2021, and 2022.
        '''
        
        r = mu_nox/Y_soc*B*(C_nox)/(C_nox+K_nox)*(1-ne)*rho_s/ne
        return r 
    @jit
    def r_act(phi,k_act,IB):
        r = phi*k_act*IB
        return r
    @jit
    def r_dact(phi,k_act,B):
        r = (1-phi)*k_act*B
        return r
    
    @jit
    def phi(C_thresh,st, C_no3):
        return 1/(jnp.exp((C_thresh-C_no3)/(st*C_thresh))+1)


    @jit
    def monod_derivatives(t,y,deltax,
                    D,
                    alpha_l,
                    v,
                    c_in,
                    no3_doc_args,
                    no3_soc_args,
                    
                    no2_doc_args,
                    no2_soc_args,
                    
                    B_args,
                    k_hyd,
                    ne,
                    rho_s,
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
        r = jnp.zeros(shape)
        y = jnp.reshape(y, shape)
        
        
        
        #Boundary condition: Inflow (advection)+ nodiffusion boundary (dispersion):
        r = r.at[0,:-2].set(-v*(y[0,:-2]-c_in)/(deltax/2) + (-mixf)*y[0,:-2] + mixf*y[1,:-2])
        
        
        kernel = np.array([mixf, -2*mixf, mixf])
        kernel = np.transpose(kernel)
        kernel = jnp.array(kernel)
       

        #Finite differences dispersion for the middle portion:
        for i in range(shape[1]-2):
            r = r.at[1:-1,i].set(jnp.convolve(y[:,i], kernel[i,:], mode = 'valid'))

        #Update the last cell (No diffusion boundary):
        r = r.at[-1,:-2].set(mixf*y[-2-1,:-2] + (-mixf)*y[-2,:-2])
        
        #Advection finite differences formulation
        r = r.at[1:-1,:-2].add(-v*(jnp.roll(y,-1,axis=0)[1:-1,:-2]-jnp.roll(y,1,axis=0)[1:-1,:-2])/(2*deltax))
        
        #Advective boundary: last term is the outflow concentration:
        r = r.at[-1,:-2].add(-v*(y[-1,:-2]-y[-2,:-2])/deltax)
        
        
        
        
        
        ## Defining the reaction terms:
        #Unpacking Parameters
        mu_no3,Y_lac,K_no3,K_lac = no3_doc_args
        mu_no3_s,Y_soc,K_no3_s = no3_soc_args
        
        mu_no2,K_no2 = no2_doc_args
        mu_no2_s,K_no2_s = no2_soc_args
        
        
        k_act,C_thresh,st = B_args
        

        #Defining reaction rates:
        ##Lactate
        r_lac_no3 = r_nox_doc(mu_no3,Y_lac,K_no3,K_lac,ne,rho_s,y[:,0],y[:,3],y[:,4])
        r_lac_no2 = r_nox_doc(mu_no2,Y_lac,K_no2,K_lac,ne,rho_s,y[:,1],y[:,3],y[:,4])
        
        ##DOC
        r_doc_no3 = r_nox_doc(mu_no3,Y_lac,K_no3,K_lac,ne,rho_s,y[:,0],y[:,2],y[:,4])
        r_doc_no2 = r_nox_doc(mu_no2,Y_lac,K_no2,K_lac,ne,rho_s,y[:,1],y[:,2],y[:,4])
        
        ##SOC
        r_soc_no3 = r_nox_soc(mu_no3_s,Y_soc,K_no3_s,ne,rho_s,y[:,0],y[:,4])
        r_soc_no2 = r_nox_soc(mu_no2_s,Y_soc,K_no2_s,ne,rho_s,y[:,1],y[:,4])
        
        ##Hydrolosis
        r_hyd = k_hyd*y[:,3]
        ##bacs
        Phi = phi(C_thresh,st, y[:,0])
       

        #r_lac_no3, r_lac_no2, r_doc_no3, r_doc_no2, r_soc_no3, r_soc_no2, r_hyd = [0,0,0,0,0,0,0]


        r = r.at[:,0].add(- 2*(r_lac_no3+r_doc_no3+r_soc_no3))
        r = r.at[:,1].add(2*(r_lac_no3+r_doc_no3+r_soc_no3) - (4/3)*(r_lac_no2+r_doc_no2) - 2*r_soc_no2)
        r = r.at[:,2].add( r_hyd-r_doc_no3 - r_doc_no2)
        r = r.at[:,3].add(- r_lac_no3 - r_lac_no2)
        r = r.at[:,4].add( r_act(Phi,k_act,y[:,5]) - r_dact(Phi,k_act,y[:,4]))
        r = r.at[:,5].add(- r_act(Phi,k_act,y[:,5]) + r_dact(Phi,k_act,y[:,4]))
        
        r = r.flatten()
        return r
    
    @jit
    def jacobian(t,y,deltax,D, alpha_l,v,c_in,no3_doc_args,no3_soc_args,no2_doc_args,no2_soc_args,B_args,k_hyd,ne,rho_s,shape):
        jac = jax.jacobian(monod_derivatives, argnums = 1)
        return jac(t,y,deltax,D,alpha_l,v,c_in,no3_doc_args,no3_soc_args,            
                        no2_doc_args,no2_soc_args,B_args,k_hyd,ne,rho_s,shape
                        )
    
    
    def calculate_stress_period(time_intervals,y0, f, f_args,jac_fun,
                                **kwargs):
        
        solution = scipy.integrate.odeint(f,y0,t = time_intervals,args = f_args, Dfun= jacobian, tfirst = True,
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
    no3_doc_args = parameters['no3_doc_args']
    no3_soc_args = parameters['no3_soc_args']
    no2_doc_args = parameters['no2_doc_args']
    no2_soc_args = parameters['no2_soc_args']
    B_args = parameters['B_args']
    k_hyd = parameters['k_hyd']
    ne = parameters['ne'] 
    rho_s = parameters['rho_s']

    #Defining the coordinate system and initial values
    deltax = length/ncells
    #xcoords = np.arange(deltax/2,length, deltax)
    B_ini = 3.3e8
    y0 = y0 = jnp.repeat(1e-11,(ncells)*6).reshape((ncells,6))
    y0.at[:,4].set(jnp.repeat(0.01*B_ini, ncells))
    y0.at[:,5].set(jnp.repeat(0.99*B_ini, ncells))
    
    shape = y0.shape
    
    y0 = y0.flatten()
    
    solutions = np.zeros((len(time_intervals),6))
    
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
                  no3_doc_args,
                  no3_soc_args,
                  no2_doc_args,
                  no2_soc_args,
                  B_args,
                  k_hyd,
                  ne,
                  rho_s,
                  shape)
        
        print(jacobian(0,y0,deltax,D, alpha_l,v,c_in,no3_doc_args,no3_soc_args,no2_doc_args,no2_soc_args,B_args,k_hyd,ne,rho_s,shape))
                    
        sol = calculate_stress_period(stress_intervals,y0 = y0, f= monod_derivatives,
                                f_args  = f_args,jac_fun = jacobian, **kwargs)
        y0 = sol[-1,:]
        
        sol = sol.reshape((sol.shape[0],shape[0],shape[1]))
 
        solutions[np.in1d(time_intervals,t_intervals),:] = sol[:,-1,:]
    
    return solutions
        

if '__int__' == '__main__':
    
    pass
    
    
    
    
    
    
    
    
    
