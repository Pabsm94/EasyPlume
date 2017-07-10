# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:07:39 2016

@author: pablo
"""

import numpy as np 

import abc

import matplotlib.pyplot as plt 

class Hyperplume():
    
    """ Parent class Hyperplume loads target plasma and defines common attributes as well as
    shared methods in the AEM and SSM plume classes"""
    
    __metaclass__= abc.ABCMeta # Python decorator used to define abstract methods at any location in the class
    
    @abc.abstractclassmethod # Defining abstract method
    
    def solver(self):
        
        """Solver Abstract Method to be particularised by each Plume code. It is only defined for
        structure purposes in parent class Hyperplume"""
        
        return 
        
    @abc.abstractclassmethod
    
    def query(self,z,r):
        
        """Query abstract method returns plasma profile data at specified grid points. query method is
        to be particularised by each plume code.It is only defined forstructure purposes 
        in parent class Hyperplume"""
        
        return 
         
    def __init__(self,plasma={'Electrons': {'Gamma': 1,'T_0_electron': 2.1801714e-19,'q_electron': -1.6e-19},'Ions': {'mass_ion': 2.1801714e-25, 'q_ion': 1.6e-19}},z_span=np.linspace(0,100,500),r_span=np.linspace(0,40,500),n_init=0.0472*np.linspace(1,0,500)**2):
        
        """ plume_constructor loads common class properties for AEM and SSM plume classes
        
        Args:
            plasma (dict): simple_plasma object dictionary containing basic plasma parameters.
            z_span (numpy.ndarray): axial region where the problem will be integrated.
            r_span (numpy.ndarray): initial far-field plasma radial profile.
            n_init (numpy.ndarray): initial dimensional density front.
        
        Usage:
            >>> Plasma = {'Electrons': {'Gamma': 1,'T_0_electron': 2.1801714e-19,'q_electron': -1.6e-19},'Ions': {'mass_ion': 2.1801714e-25, 'q_ion': 1.6e-19}}
            >>> z_span = np.linspace(0,100,100)
            >>> r0 = np.linspace(0,3,100)
            >>> n0 = np.exp(-6.15/2*r_span**2)
            >>> Plume = Hyperplume(Plasma,z_span,r0,n0)
        """
        
        self.plasma = plasma
        self.Gamma = plasma['Electrons']['Gamma']
        self.T_0 = plasma['Electrons']['T_0_electron']
        self.m_ion = plasma['Ions']['mass_ion']
        self.q_ion = plasma['Ions']['q_ion']
        self.z_span = z_span
        self.eta = r_span
        self.n0 = n_init
        
        
    def simple_plasma(self,charge=1.6e-19,ion_mass=2.1801714e-25,init_plasma_temp=2.1801714e-19,Gamma=1):
        
        """ Method simple_plasma allows the user to quickly create a Plasma dictionary with two particle species (ions and electrons), 
        and well defined attributes.
        
        Args:
            charge (float): Electron charge given dimensional in units [C]
            ion_mass(float): Ion mass given in dimensional units [Kg]
            init_plasma_temp(float): Initial plasma temperature given in dimensional units [J]
            Gamma(int or float): Dimensionless thermal expansion constant. Must be inside isothermal and polytropic boundaries [1,5/3]
        
        Returns:
            plasma (dict): Dictionary containing two simple plasma species (ions and electrons) with the before mentioned
                       properties stored in favorable form 
        Usage:             
            >>> Plasma = Hyperplume().simple_plasma(charge=1.6e-19,ion_mass=2.1801714e-25,init_plasma_temp=2.1801714e-19,Gamma=1)
            
        """
        
        if Gamma  < 1 or Gamma > 2: #checking thermal expansion model
            
            print ('Gamma is outside isothermal or polytropic boundaries')
            
        else:
            
            plasma={'Ions':{'mass_ion': ion_mass,'q_ion':charge}, 'Electrons':{'q_electron': -charge,'T_0_electron':init_plasma_temp,'Gamma':Gamma} }
            
            return plasma 
        
    def temp(self,n,n_0,T_0,Gamma):
        
        """ Method temp calculates plasma temperature (T) as function of plasma density (n)
        
        Args:
            n(int or np.ndarray): plasma density at specific (z,r) location in the plume grid
            n_0 (int):Iinitial density of plasma
            T_0 (float): Initial temperature of plasma
            Gamma (int): Dimensionless thermal expansion constant
        
        Returns:
            T (float or np.ndarray): Temperature of plasma at targeted (z,r) grid points in plume
            
        Usage:
            >>> T = Hyperplume().temp(n=0.65,n_0=1,T_0=2.1801714e-19,Gamma=1)
        
        """
        
        if Gamma == 1: #Checking expansion model
            
            T = T_0*(n*0 + 1)
            
        else:
            
            T = T_0*((n/n_0)**(Gamma-1))
            
        return T
            
    
    def phi (self,n,n_0,T_0,Gamma,e_charge):
        
        """Method phi calculates electric potential (\phi) as function of plasma density (n)
        
         Args:
             n(int or np.ndarray): plasma density at specific (z,r) location in the plume grid
             n_0 (int):Iinitial density of plasma
             T_0 (float): Initial temperature of plasma
             Gamma (int): Dimensionless thermal expansion constant
             e_charge (float):Electron charge
             
         Returns:
             phi(float or np.ndarray): Electric potential of plasma at (z,r) targeted grid point
             
         Usage:
             >>> phi = Hyperplume().phi(n=0.65,n_0=1,T_0=2.1801714e-19,Gamma=1,e_charge=-1.6e-19)
             
         """
        
        if Gamma == 1: #Checking expansion model
        
            phi =  (T_0/e_charge)*np.log(n/n_0)
        
        else :
            
            phi =  (T_0/e_charge)*(Gamma / ((Gamma - 1)) * ((n/n_0)**(Gamma-1)-1))
            
        return phi
        
    def n(self,n_0,T_0,phi,Gamma,e_charge):
        
        """Method n calculates plasma density (n) as function of plasma potential (\phi)
        
         Args:
             n_0 (int):Iinitial density of plasma
             T_0 (float): Initial temperature of plasma
             Gamma (int): Dimensionless thermal expansion constant
             e_charge (float):Electron charge
         Returns:
             n (float or numpy.ndarray): Pasma density at (z,r) targeted grid point in the plume.
             
         Usage:
             n = Hyperplume.n(n_0=1,T_0=2.1801714e-19,phi=-5.7,Gamma=1,e_charge=-1.6e-19)
         
         """
        
            
        
        if Gamma == 1: #Checking expansion model
            
            n =  n_0*np.exp(phi*e_charge/T_0)
            
        else:
            
            n = n_0*(((Gamma-1)/Gamma*phi*e_charge/T_0 + 1 )**1/(Gamma-1))
            
        return n
        
    def eta_deriver(self,x,y):
        
        """Method eta_derivar calculates the numerical derivatives of the variables along eta, with a

        Args: 
            x (np.ndarray): represents the derivative step (dx,dy)
            y (np.ndarray): vector to derive with respect to x
        
        Returns:
            y_prime(np.ndarray): derivaive of y over x  stored in array format
            
        Usage:
            >>> x = np.array([0,0.5,1,1.2,2,2.3,2.6])
            >>> y = np.array([10,17,23,27,36,40,45])
            >>> dydx = Hyperplume.eta_deriver(x,y)
        """
        
        dx = np.gradient(x)
        
        y_prime = np.gradient(y,dx)
        
        return  y_prime
        
    def plot(self,z=np.array([15,20,25,30]),r=np.array([20,25,30,35]),var_name='n',contour_levels=[0,1,2,3,4,5,6,7,8]):
        
        """ Hyperplume Class method to plot the contours of important plasma variables along the specified (z,r) plume grid points
        
        Args:
        
            z (int,float, or np.ndarray): new interpolation axial region where plasma variabes are to be calculated and plotted. Must be inside z_grid limits
            r (int,float, or np.ndarray): new interpolation axial region where plasma variabes are to be calculated and plotted. Must be inside z_grid limits   
            var_name (str): string containing the name of the variable to be visualized. Options are:
                            'lnn': logarithm of plasma density 
                            'u_z': axial plume velocity
                            'u_r':radial plume velocity
                            'T': plasmaTemperature
                            'phi':  ambipolar electric field
                            'eta': ion stream lines
            contour_levels (array or of list): contour lables of plasma varialbled at the targets z,r points.
            
        Returns:
            None
            
        Usage:
            >>> Plasma = Hyperplume().SIMPLE_plasma()
            >>> Plume = AEM()
              
        """
        
        lnn,u_z,u_r,T,phi,error,eta = self.query(z,r) #Retrievibg plasma variables at z,r gid points 
    
        fig = plt.figure()
            
        CE = plt.contour(z,r,eval(var_name),contour_levels)  
        plt.title(var_name)
        plt.xlabel(r'$\ z/R_0 $')
        plt.ylabel(r'$\ r/R_0 $')
        plt.ylim(0,10)
        plt.clabel(CE,CE.levels,fontsize=6)
        
        plt.savefig(var_name + '.pdf',bbox_inches='tight')
        
        fig.show()


        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    