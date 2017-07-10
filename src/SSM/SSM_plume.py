# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 13:32:28 2016

@author: pablo
"""

import os 
    
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    
import sys
 
sys.path.append(dir_path) #path to src

from src import np,odeint,interp1d,Hyperplume,plt


class SSM(Hyperplume):
    
    """Self Similar model of a plasma plume expansion.Class SSM inherits methods __init__,solver 
    and query from parent class Hyperplume, and particularizes them. """
    
    
    
    def __init__(self,plasma={'Electrons': {'Gamma': 1,'T_0_electron': 2.1801714e-19,'q_electron': -1.6e-19},'Ions': {'mass_ion': 2.1801714e-25, 'q_ion': 1.6e-19}},M_0=40,d_0=0.2,z_span=np.linspace(0,100,500),r_span=np.linspace(0,40,500),n_init=0.0472*np.linspace(1,0,500)**2):
        
        """Constructor __init__ loads and initialises the main class attributes.
        Calls parent class Hyperplume constructor method __init__ to store main plasma properties as attributes in the class.
        
        Args:
            plasma (dict): simple_plasma object dictionary containing basic plasma parameters.
            z_span (numpy.ndarray): axial region where the problem will be integrated.
            r_span (numpy.ndarray): initial far-field plasma radial profile.
            n_init (numpy.ndarray): initial dimensional density front.
            M_0 (float): Plasma Mach number at (z,r) = (0,0)
            d_0 (float): Tangent of initial plume divergence angle
            
        Implementation:
            >>> e_charge,ion_mass,Plasma_temp,gamma_value=1.6e-19,2.1801714e-25,2.1801714e-19,1 #Main Plasma Parameters
            >>> Plasma = Hyperplume().simple_plasma(e_charge,ion_mass,Plasma_temp,gamma_value) #Loading Plasma dict
            >>> z_span = np.linspace(0,110,5000) # Axial plume grid for integration
            >>> r_0 = np.linspace(0,10,5000) #Initial plume radial profile
            >>> n0 =  np.exp(-6.15/2 * r_0**2) #Initial far-field plume density
            >>> M0,d0=20,0.2 #SSM parameters: Mach number and initial far-field plume diveregence
            >>> Plume = SSM(Plasma,M0,d0,z_span,r_0,n0) #Creating SS plume
                
        """
        
        super(SSM,self).__init__(plasma,z_span,r_span,n_init)
        
        self.M_0,self.d_0 = M_0,d_0 #Loading Mach, and initial divergece. Variables inherent to SSM Plume
        
    def solver(self):
        
        """Solver method solves for model constriants C and h, as well as  the initial dimensionless axial velocity vector upsilon 
        and initial dimensionless density profile nu, using SSM model equations. It then saves this plume variables as as class attributes,
        in the form of interpolation libraries over the entire plume grid.
        
        Solver method is a particularization of the abstrac Hyperplume.solver() method
        
        Implementation:
            
            >>> Plume.solver() # be sure to create a valid SSM plasma plume before applying the plume solver method
        
        To access the interpolation libraries and SSM constraints particularly:
            >>> print(Plume.C,Plume.h) #SSM model constraints
            >>> Plume.nu_interp #Initial dimensionless density interpolation library
            >>> Plume.upsilon_interp #Initial dimensionless axial velocity interpolation library
        """
        
        nu = self.n0/self.n0[0] #Dimensionles initial density front
        
        nu_prime = np.empty(nu.size) #Derivative of initial dimensionless density front. Needed for upsilon calculations
            
        nu_prime = self.eta_deriver(self.eta,nu) #Call to superclass Hyperplume() method self.eta_deriver(x,y)
        
        nu_prime[0] = 0 # Edge array conditions for the derivative  of the density front
            
        nu_prime[-1] = (nu[-1] - nu[-2]) / (self.eta[-1] - self.eta[-2])
        
        self.C =  -2*(nu[1] - nu[0]) / (self.eta[1] - self.eta[0])**2 #Scaling Separtion Constant of SSM Model
        
        upsilon = np.sqrt(-nu**(self.Gamma-2) * nu_prime / (self.eta * self.C)) #Dimensionles initial axial velocity front
        
        upsilon[0] = 1
        
        def dh_fun(h,Z):
            
            """dh_fun function calculates the derivative of the self-similar dilation function h(z), 
            and saves the results as a class attribute in column-array format 
            
            Args:
                h (numpy.ndarray): SSM model scaling function
                Z (numpy.ndarray): axial span for integration. Coincidet with initial axial span loaded in SSS 
                                                                class constructor,for accruacy and correctness)
            Returns:
                df (numpy.ndarray): derivative of SSM scaling function
                
            """
            
            "Checking thermal expanion model (isothermal or polytropic coesfficient)"
            
            if self.Gamma == 1:
            
                dh = np.sqrt(self.d_0**2 + (self.C/self.M_0**2) * 2 * np.log(h))
          
            else: 
        
                dh = np.sqrt(self.d_0**2 + (self.C/self.M_0**2) * -(h**(2-2*self.Gamma) -1) * 1/(self.Gamma-1))
        
            return dh
        
        h_init = 1
        
        h = odeint(dh_fun,h_init, self.z_span) # solves numerically the ODE in dh_fun, to obtain 
      
        dh = dh_fun(h, self.z_span) # Call fun dh_fun to solve for dh
               
        """Creation of 1D interpolation libraries for the main attributes, to be used later in query method at 
        the targeted (Z,r) in the plume."""
    
        self.h = np.reshape(h,self.z_span.size) #MMM20170424: what is this? what for? 
        
        #PABLO20170424: output h from odeint (line 108) is a column array h.shape = (). 
        #Here, it is reshaped in proper form for python interp1d method (next line)
        
        self.h_interp = interp1d(self.z_span,self.h,kind='linear') #Creating interpolation library of self-similarity h(z) function
        
        self.dh = np.reshape(dh,self.z_span.size) #PABLO20170424:same as before
        
        self.dh_interp = interp1d(self.z_span,self.dh,kind = 'linear')
        
        self.nu_interp = interp1d(self.eta,nu,kind = 'linear') #Creating interpolation library of self-similarity derivative dh(z) function
        
        self.nu_prime_interp = interp1d(self.eta,nu_prime,kind = 'linear') #Creating interpolation library of dimensionless initial density
        
        self.upsilon_interp = interp1d(self.eta,upsilon,kind='linear') #Creating interpolation library of dimensionless initial axial velocity
        
    def query(self,z,r):
        
        """ Method query returns the density, velocity profile, temperature, the electric potential and SSM error at
        particular (z,r) points by interpolation over the Plume grid.
        SSM method query is a particulatization of the abstract Hyperplume method Hyperplume.query()
        
        Args:
            z (int,numpy.ndarray): axial target points where plasma variables are retrieved. Single points, arrays of locations and meshgrids are valid.
            r (int,numpy.ndarray): axial target points where plasma variables are retrieved. Single points, arrays of locations and meshgrids are valid.
        
        Returns:
            lnn (int,numpy.ndarray): logarithmic plasma density at specified (z,r) points in plume grid
            u_z (int,numpy.ndarray): plasma axial velocity at specified (z,r) points in plume grid
            u_r (int,numpy.ndarray): plasma radial velocity at specified (z,r) points in plume grid
            T (int,numpy.ndarray): plasma temperature at specified (z,r) points in plume grid
            phi (int,numpy.ndarray): plasma ambipolar electric potential at specified (z,r) points in plume grid
            error (int,numpy.ndarray): SSM error created by imposing model constraints at specified (z,r) points in plume grid
            eta (int,numpy.ndarray): ion current stream lines at specified (z,r) points in plume grid
            
        Usage:
            >>> z,r = np.linspace(0,100,50),np.linspace(0,50,40) #target (z,r) for plume study
            >>> lnn,u_z,u_r,T,phi,error,eta=Plume.query(z,r)
            """
        
        eta = r/self.h_interp(z) #calculation of eta at user targeted grid point
        
        n = self.n0[0] * self.nu_interp(eta) * 1/self.h_interp(z)**2 #Dimensional density at targetd (z,r) points
        
        lnn = np.log(n)
        
        #Calling various Hyperplume methods to calculate remaining plasma parameters based on plume density
        
        T = self.temp(n,self.n0[0],self.T_0,self.Gamma) #Dimensional Temperature at targetd (z,r) points
        
        phi = self.phi(n,self.n0[0],self.T_0,self.Gamma,self.q_ion) #Dimensional potential at targetd (z,r) points
        
        u_z = self.M_0*np.sqrt(self.Gamma*self.T_0/self.m_ion) * self.upsilon_interp(eta) #Dimensional axial velocity at targetd (z,r) points
        
        u_r = self.d_0 * u_z * self.dh_interp(z) * eta #Dimensional radial velocity at targetd (z,r) points
        
        error = self.C * self.dh_interp(z) / (self.M_0**2 * (self.h_interp(z)**(2*self.Gamma-1))) * (4 * eta * self.nu_interp(eta) / self.nu_prime_interp(eta) + 2 * eta**2) #SSM error at targetd (z,r) points
           
        return lnn,u_z,u_r,T,phi,error,eta
       
    
# Helper functions 
            
def type_parks(plasma={'Electrons': {'Gamma': 1,'T_0_electron': 2.1801714e-19,'q_electron': -1.6e-19},'Ions': {'mass_ion': 2.1801714e-25, 'q_ion': 1.6e-19}},M_0=40,d_0=0.2,z_span=np.linspace(0,100,500),r_0=np.linspace(0,40,500),C=-2*np.log(0.05)):
    
    """ type_parks functions allow the user to generate default plume density profiles based on the theoretical 
    Parks plume model. The function creates the initial density profile following
    the theoretical model, and creates a SSM Plume object with unique characteristics
    
    Args:
        plasma (dict): Hyperplume's simple_plasma object, or otherwise a similar plasma dictionary containing basic parameters.
        z_span (numpy.ndarray): axial region where the problem will be integrated.
        r_0 (numpy.ndarray): initial far-field plasma radial profile.
        M_0 (float): Plasma Mach number at (z,r) = (0,0)
        d_0 (float): Tangent of initial plume divergence angle
        C (float): SSM model constraint. C is a separation constant used for scaling the Self-Similarity plume proble.
                   C is used to determine the initial density profile derived by Parks. In particular:
                   n_parks = np.exp(-C*r_0**2 /2)
   Returns:
       Plume (object): SSM Plume object preloaded and solved with Parks theoretical density and axial velocity models.
       
   Usage:
       >>> Plasma = Hyperplume().simple_plasma(e_charge,ion_mass,Plasma_temp,gamma_value) #Loading Plasma dict
       >>> z_span = np.linspace(0,110,5000) # Axial plume grid for integration
       >>> r_0 = np.linspace(0,10,5000) #Initial plume radial profile
       >>> C = 6.15
       >>>Plume_parks = type_parks(Plasma,z_span,r_0,C)
       >>> lnn,u_z_,u_r,T,phi,error,eta=Plume_parks.query(z,r)
        
        """

    if plasma['Electrons']['Gamma'] is not 1:
        
        print ('Gamma must be 1 for Parks model')
        
    else:
        
        n0 = np.exp(-C * r_0**2 /2) # fixing plume initial far-region density profile following Parks model
        
        Plume = SSM(plasma,M_0,d_0,z_span,r_0,n0) # creating SSM plume
        
        Plume.solver() # solving plume with general model equations
        
        Plume.upsilon = np.ones(Plume.n0.shape) # fixing plume initial far-region axial velocity profile following Parks model
        
        _,_,_,_,_,_,eta = Plume.query(z_span,r_0) #calculating eta values following general model equations
        
        Plume.upsilon_interp = interp1d(eta,Plume.upsilon,kind='linear') # interpolating in fixed parks inital velocity profile
        
        return Plume

def type_korsun(plasma={'Electrons': {'Gamma': 1,'T_0_electron': 2.1801714e-19,'q_electron': -1.6e-19},'Ions': {'mass_ion': 2.1801714e-25, 'q_ion': 1.6e-19}},M_0=40,d_0=0.2,z_span=np.linspace(0,100,500),r_0=np.linspace(0,40,500),C = 2*(0.05**(-2/1.3)-1)):
       
    """ type_parks functions allow the user to generate default plume density profiles based on the theoretical 
    Korsun plume model. The function creates the initial density profile following
    the theoretical model, and creates a SSM Plume object with unique characteristics
    
    Args:
        plasma (dict): Hyperplume's simple_plasma object, or otherwise a similar plasma dictionary containing basic parameters.
        z_span (numpy.ndarray): axial region where the problem will be integrated.
        r_0 (numpy.ndarray): initial far-field plasma radial profile.
        M_0 (float): Plasma Mach number at (z,r) = (0,0)
        d_0 (float): Tangent of initial plume divergence angle
        C (float): SSM model constraint. C is a separation constant used for scaling the Self-Similarity plume problem.
                   C is used to determine the initial density and axial velocity profiles derived by Korsun. In particular:
                   n_parks = 1 / (1 + C / 2 * eta_0**2 )
                   upsilon_parks = (1 + C / 2 * eta_0**2 )**(gamma/2)
                   
   Returns:
       Plume (object): SSM Plume object preloaded and solved with Parks theoretical density and axial velocity models.
       
   Usage:
       >>> Plasma = Hyperplume().simple_plasma(e_charge,ion_mass,Plasma_temp,gamma_value) #Loading Plasma dict
       >>> z_span = np.linspace(0,110,5000) # Axial plume grid for integration
       >>> r_0 = np.linspace(0,10,5000) #Initial plume radial profile
       >>> C = 6.15
       >>>Plume_parks = type_parks(Plasma,z_span,r_0,C)
       >>> lnn,u_z_,u_r,T,phi,error,eta=Plume_parks.query(z,r)
        
    """
    n0 = 1 / (1 + C / 2 * r_0**2 ) # fixing plume initial far-region density profile following Korsun model
    
    Plume = SSM(plasma,M_0,d_0,z_span,r_0,n0) # creating SSM plume
        
    Plume.solver() # solving plume with general model equations
        
    Plume.upsilon = (1 + C / 2 * r_0**2 )**(-Plume.Gamma/2) # fixing plume initial far-region axial velocity profile following Korsun model
    
    _,_,_,_,_,_,eta = Plume.query(z_span,r_0) # retrieving eta values following general model equations
        
    Plume.upsilon_interp = interp1d(eta,Plume.upsilon,kind='linear')  # interpolating in fixed Korsun inital velocity profile
    
    return Plume
    
def type_ashkenazy(plasma={'Electrons': {'Gamma': 1,'T_0_electron': 2.1801714e-19,'q_electron': -1.6e-19},'Ions': {'mass_ion': 2.1801714e-25, 'q_ion': 1.6e-19}},M_0=40,d_0=0.2,z_span=np.linspace(0,100,500),r_0=np.linspace(0,40,500),C = 0.2**2*(1-2*np.log(0.01)/np.log(1+0.2**2))):
    
    """ type_ashkenazy functions allow the user to generate default plume density profiles based on the theoretical 
    ashkenazy plume model. The function creates the initial density profile following
    the theoretical model, and creates a SSM Plume object with unique characteristics
    
    Args:
        plasma (dict): Hyperplume's simple_plasma object, or otherwise a similar plasma dictionary containing basic parameters.
        z_span (numpy.ndarray): axial region where the problem will be integrated.
        r_0 (numpy.ndarray): initial far-field plasma radial profile.
        M_0 (float): Plasma Mach number at (z,r) = (0,0)
        d_0 (float): Tangent of initial plume divergence angle
        C (float): SSM model constraint. C is a separation constant used for scaling the Self-Similarity plume problem.
                   C is used to determine the initial density profile derived by Ashkenazy. In particular:
                   n_parks = (1 + k*eta_0**2)**(-C/(2*k))
                   upsilon_parks = (1 + k*eta_0**2)**(-1/2), where k = d_0**2
                   
   Returns:
       Plume (object): SSM Plume object preloaded and solved with Parks theoretical density and axial velocity models.
       
   Usage:
       >>> Plasma = Hyperplume().simple_plasma(e_charge,ion_mass,Plasma_temp,gamma_value) #Loading Plasma dict
       >>> z_span = np.linspace(0,110,5000) # Axial plume grid for integration
       >>> r_0 = np.linspace(0,10,5000) #Initial plume radial profile
       >>> C = 6.15
       >>>Plume_parks = type_parks(Plasma,z_span,r_0,C)
       >>> lnn,u_z_,u_r,T,phi,error,eta=Plume_parks.query(z,r)
        
    """
    if plasma['Electrons']['Gamma'] is not 1:
        
        print('Gamma must be 1 for Ashkenazy model')
        
    else:
        
        k = d_0**2
        
        n0 = (1 + k*r_0**2)**(-C/(2*k)) # fixing plume initial far-region density profile following Ashkenazy model
        
        Plume = SSM(plasma,M_0,d_0,z_span,r_0,n0) # creating SSM plume
        
        Plume.solver() # solving plume with general model equations
        
        Plume.upsilon = (1 + k*r_0**2)**(-1/2) # fixing plume initial far-region axial velocity profile following Ashkenazy model
        
        _,_,_,_,_,_,eta = Plume.query(z_span,r_0) # retrieving eta values following general model equations
        
        Plume.upsilon_interp = interp1d(eta,Plume.upsilon,kind='linear') # interpolating in fixed Korsun inital velocity profile
        
        return Plume        
    

       