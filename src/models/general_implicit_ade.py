# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:43:47 2022

@author: Vitor Cantarella


1D - column model


The script discretize the model in cells and calculate the advection-dispersion-reaction solution
for each substance of interest using an implicit finite difference scheme.

Some limitations:
    flow velocity is either constant or specified. This mean, that for now it does not calculate groundwater flow
    It is also built to be easily appended with additional substances.

Original work is to model Nitrate reaction processes in the aquifer.
"""
import scipy
import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)


class ColumnModel():
    """
    Object to store the column model details and calculate the results
    """
    
    def __init__(self, subs_dictionary, transport_indicator: np.ndarray):
        """
        1D Reactive Tranport model.
        Each substance that is explicit modelled must be included. This means also bacteria cells or enzymes, which may be stuck to the matrix,
        must also be included,
        
        In adsorption studies, the adsorbed phase must be included.

        Parameters
        ----------
        subs_dictionary : key: substance index, value: substance name
            DESCRIPTION.
        transport_indicator: np.ndarray
            each row of the array represents a substance, input 1 for transported and 0 for non transported

        Returns
        -------
        None.

        """
        self.subs_dictionary = subs_dictionary
        self.transport_indicator = transport_indicator
        self.kinetic_expressions = {k: None for k, v in subs_dictionary.items()}
        self.kinetic_parameters = {k: None for k, v in subs_dictionary.items()}
    
    def add_transport_parameters(self, velocity: np.ndarray,
                                 longitudinal_dispersivity: np.ndarray,
                                 diffusion_coefficient: np.ndarray):
        """
        Add the transport parameters to the model. They can be either a scalar (in numpy array format) or an array.
        If array is given, it must be compatible with the grid information.

        Parameters
        ----------
        velocity : np.ndarray
            can be an array in case of a defined grid
        longitudinal_dispersivity : np.ndarray
            can be an array in case of a defined grid
        diffusion_coefficient : np.ndarray
            This is defined for each transported substance or the same for all

        Returns
        -------
        None.

        """
        self.velocity = velocity
        self.longitudinal_dispersivity = longitudinal_dispersivity
        self.diffusion_coefficient = diffusion_coefficient
    
    def add_kinetic_expression(self, f, parameters, substance_index: int):
        """
        

        Parameters
        ----------
        f : function that calculates the kinetic rate law for the substance
            the function has to be defined as f(y, parameters), with y being the matrix of concentrations for each substance
        parameters : tuple
            the parameters used as input in the function f
        substance_index : int
            substance index to which the rate will be applied.

        Returns
        -------
        None.

        """
        self.kinetic_expressions[substance_index] = jit(f)
        self.kinetic_parameters[substance_index] = parameters
    
    def add_grid(self, length, ncells=None, deltax=None, coords=None):
        """
        Add model grid (optional)

        Parameters
        ----------
        length : float
            1D column length
        ncells: float
            number of cells, must be compatible with deltax.
        deltax : np.ndarray or float
            deltax of each cell (if given). Can be an array specifiying the length of each cell of a float specifying uniform spacing
        coords : TYPE, optional
            x-coords of each cell center.
        

        Returns
        -------
        None.

        """
        self.length = length
        self.ncells = ncells
        self.deltax = deltax
        self.coord = coords
    
    def initial_values(self, y0:np.ndarray):
        """
        

        Parameters
        ----------
        y0 : np.ndarray
            Array of initial values for the model. The rowsize must be the same as the grid coordinates.
            The cols are the substances index

        Returns
        -------
        None.

        """
        
        self.y0 = y0
    def stress_periods(self,start_time_array: np.ndarray, inflow_concentration_array: np.ndarray):
        """
        Add the stress period information to the model

        Parameters
        ----------
        start_time_array : np.ndarray
            start time of each stress period
        inflow_concentration_array : np.ndarray
            inflow concentration matrix where each row represents a stress period and each column a substance (use any_value for non transported substance)

        Returns
        -------
        None.

        """
        self.start_time_array = start_time_array
        self.inflow_concentration_array = inflow_concentration_array
    @partial(jit, static_argnums=(0,))    
    def transport_equation(self,y, c_in):
        """
        Internal formulation of the coupled advection-transport equation scheme.

        """
        
        #initializing variables
        deltax = self.deltax
        D = self.diffusion_coefficient
        alpha_l = self.longitudinal_dispersivity
        v = self.velocity
        
        dcdt = jnp.zeros(y.shape[0])
        
        #upwind for advection calculation
        c_up = jnp.concatenate([c_in,y])
        
        #advection step
        advec = jnp.gradient(c_up, deltax)*(-v)
        dcdt = advec[1:]
        
        #dispersion step
        diff = (D + v*alpha_l)*jnp.gradient(jnp.gradient(y,deltax),deltax)
        
        #Adding dispersion to advection
        dcdt = dcdt.at[:].add(diff)
    
        return dcdt
    #@partial(jit, static_argnums=(0))
    def model_build(self,t,y,c_in):
        #print(y)
        y = jnp.reshape(y, (self.ncells,len(self.subs_dictionary)),order = 'F')
        
        dcdt_l = []
        for j, value in self.subs_dictionary.items():
            c_j = np.array([c_in[j]])
            #print(c_j)
            
            dcdt_j = self.transport_equation(y[:,j].squeeze(),c_j)*self.transport_indicator[j]+self.kinetic_expressions[j](y, self.kinetic_parameters[j])
            #print(dcdt_j)
            dcdt_l.append(dcdt_j)
        
        dcdt = jnp.concatenate(dcdt_l)
        return dcdt
        
    #@partial(jit, static_argnums=(0,))
    def jac_fun(self,t,y,c_in):
        return jax.jacobian(self.model_build, argnums= 1)(t,y,c_in)
    
        

    
    def run_model(self, time_intervals):
        """
        Run Advetion dispersion model
        
        time_intervals: time intervals where the model should be evaluated.
        It must contain the values in start_time_array

        Returns
        -------
        concentrations 

        """
        output = []
        #Initializing concentrations:
        y0 = self.y0
        y0 = jnp.ravel(y0, order = "F")
        
        #Calculating each stress period
        for i in range(len(self.start_time_array)):
            #stress periods
            if i < len(self.start_time_array)-1:
                t_intervals = time_intervals[np.where((time_intervals >= self.start_time_array[i]) & (time_intervals <= self.start_time_array[i+1]))]
            else:
                t_intervals = time_intervals[np.where((time_intervals >= self.start_time_array[i]))]
            
            #Time interval per stress period:
            stress_intervals = t_intervals.copy() - t_intervals[0]
            
            
            #Inflow concentration must be set as an array for tranport to work:
            c_in = jnp.array(self.inflow_concentration_array[i])
            
            
            
            
            f_ode = partial(self.model_build, c_in = c_in)
            f_jac = partial(self.jac_fun, c_in = c_in)
            ode_output = scipy.integrate.solve_ivp(f_ode,t_span = (stress_intervals[0],stress_intervals[-1]),y0 = y0,method = 'BDF', t_eval = stress_intervals, jac = f_jac)
            
            y0 = jnp.array(ode_output.y.T[-1,:])
            
            
            ode_output = ode_output.y.T.reshape((stress_intervals.shape[0],self.ncells, len(self.subs_dictionary)),order = 'F')
            
            output.append(ode_output)
        
        output = np.vstack(output)
        
        return output
            
            
            
    
    