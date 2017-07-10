# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 19:12:18 2017

@author: pablo
"""

import os 

dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
 
sys.path.append(dir_path) #change to src
from src import np,math,interp1d,Hyperplume,griddata,interp2d,plt

import matplotlib.lines as mlines



class AEM(Hyperplume):
    
    """Asymptotic Expansion Model of a plasma plume expansion.Class AEM inherits methods  from 
    parent class Hyperplume, and particularizes them.All initial inputs must be given in dimensional form.
    
    """
    
    def __init__(self,plasma={'Electrons': {'Gamma': 1,'T_0_electron': 2.1801714e-19,'q_electron': -1.6e-19},'Ions': {'mass_ion': 2.1801714e-25, 'q_ion': 1.6e-19}},z_span=np.linspace(0,100,500),r_span=np.linspace(0,40,500),n_init=np.linspace(1,100,500),uz_init=np.linspace(1,100,500),ur_init=np.linspace(0,100,500),sol_order=0):
        
        """ Class method __init__ is used as class constructor. Calls parent class Hyperplume constructor method __init__ to 
        store main plasma properties as attributes in the class.
        
        Args:
            plasma (dict): simple_plasma object dictionary containing basic plasma parameters.
            z_span (numpy.ndarray): axial region where the problem will be integrated.
            r_span (numpy.ndarray): initial far-field plasma radial profile.
            n_init (numpy.ndarray): initial far-field plum density front.
            uz_init (numpy.ndarray): initial far-region plume axial velocity profile
            ur_init (numpy.ndarray): initial fr-region plume radial velocity profile
            sol_order (int): Integer defining the AEM correction order for the plume integration.
                            -0: AEM "cold beam" zeroth order solution
                            -1: AEM first order correction
                            -2: Second Order Correction
                            
        Usage:
            >>>>>> e_charge,ion_mass,Plasma_temp,gamma_value=1.6e-19,2.1801714e-25,2.1801714e-19,1 #Main Plasma Parameters
            >>> Plasma = Hyperplume().simple_plasma(e_charge,ion_mass,Plasma_temp,gamma_value) #Loading Plasma dict
            >>> z_span = np.linspace(0,110,5000) # Axial plume grid for integration
            >>> r_0 = np.linspace(0,10,5000) #Initial plume radial profile
            >>> n0 =  np.exp(-6.15/2 * r_0**2) #Initial far-field plume density
            >>> uz0,,ur0  = np.linspace(20000,20000,100),np.linspace(0,40000,100) #Intial far-field plume axial and radial velocity fronts
            >>> AEM_Order = 2 # AEM model solution
            >>> PlumeAEM = AEM(Plasma,z_span,eta_0,n0,uz0,ur0,AEM_Order) #Creation of AEM plume
        
        Other important class attributes loaded in the AEM constructor are:
            d0 (numpy.ndarray): far field initial divergence ur0/uz0.
            d0p (numpy.ndarray): derivative of plume initial divergence
            eps (float): AEM expansion parameter 1/M_{0}^{2}
            uz0p (numpy.ndarray): derivative of initial far region axial velocity
            duz0p (numpy.ndarray): derivative of initial far region radial velocity
            z_grid,r_grid (numpy.ndarray): Plume grids where AEM problem is integrated
            
        To access these attributes, for instance:
            >>> print(PlumeAEM.d0)
            >>> print(PlumeAEM.eps)
        """
        
        #Call parent class Hyperplume constructor method to store main plasma properties as attributes in the AEM class."""
        super(AEM,self).__init__(plasma,z_span,r_span,n_init)
        
        self.uz0,self.ur0,self.d0 = uz_init,ur_init,ur_init/uz_init #Load additional AEM plasma plume properties
        self.alpha0=math.degrees(math.atan(interp1d(self.eta,self.d0)(1))) #Initial Plume divergence at the 95% streamline
        self.order = sol_order #AEM Solution Order
        self.eps = self.Gamma*self.T_0/(self.m_ion*(uz_init[0]**2 + ur_init[0]**2)) #residual expansion parameter. Inverse of squared Mach Number 
        self.M0 = np.sqrt(1/self.eps) #Plume Mach Nmeber
        
        "Derivatives of initial front"
        
        self.d0p = self.eta_deriver(self.eta,self.d0) #derivative of plume divergence
        self.d0p[0],self.d0p[-1] = self.d0[1]/self.eta[1],self.d0p[-2] + (self.d0p[-2] - self.d0p[-3]) #Edge vetor prime conditions
        self.uz0p = self.eta_deriver(self.eta,self.uz0) #derivative of initial axial velocity
        self.uz0p[0],self.uz0p[-1] = 0,self.uz0p[-2] + (self.uz0p[-2] - self.uz0p[-3])
        self.duz0p = self.eta_deriver(self.eta,self.d0*self.uz0) #derivatie of initial radial velocity
        self.duz0p[0],self.duz0p[-1] = self.duz0p[1]/self.eta[1],self.duz0p[-2] + (self.duz0p[-2] - self.duz0p[-3])
        
        "Grid Set Up"
    
        self.z_grid,self.r_grid = self.grid_setup(self.z_span.size,self.eta.size) #2D grids of z, and r points in the plume
                
    def solver(self):
        
        """ Class method Solver integrates the AEM model equations in the specified plume grid. The method stores the different order
        plasma properties in matrixes of size (mxn), where m,n are the number of z,r points,respectively.Porperties such as
        density,temperature,electric field,etc are calculated and saved as attributes of the class in this matrix form.
        
        Usage:
            >>> PlumeAEM = AEM(Plasma,z_span,eta_0,n0,uz0,ur0,AEM_Order) #Creation of AEM plume
            >>> PlumeAEM.solver() # be sure to create a valid AEM plasma plume before applying the plume solver method
        
        Main Plume properties solved and saved by the method as class attributes:
            lnn (numpy.ndarray): 3-D matrix containing density values (logarithmic) for the three possible AEM solution orders.
            uz (numpy.ndarray): 3-D matrix containing axial velocity values for the three possible AEM solution orders.
            ur (numpy.ndarray): 3-D matrix containing radial velocity values for the three possible AEM solution orders.
            T (numpy.ndarray): 3-D matrix containing plasma Temperature values for the three possible AEM solution orders.
            phi (numpy.ndarray):3-D matrix containing plasma ambipolar electric field for the three possible AEM solution orders.
            div (numpy.ndarray): 3-D matrix containing plume divergence values for the three possible AEM solution orders.
            eta_ (int,numpy.ndarray): 3-D matrix containing ion current streamlines for the three possible AEM solution orders.
        
        To access these varibles,for instance:
            >>> PlumeAEM.lnn[0,:,:] #density values for Cold Beam Zeroth Order AEM solution of plume expansion in the grid
            >>> PlumeAEM.uz[1,:,:] # axial velocity First Order AEM solution
            >>> PlumeAEM.T[2,:,:] ## Temperature values for Second Order AEM solution
        """
        
        self.__zpts = np.shape(self.z_grid)[1] #Number of axial steps
        
        self.__epts = np.shape(self.r_grid)[0] #Number of radial steps
        
        self.uz = np.zeros((3,self.__epts,self.__zpts)) #3D matrix containing the axial velocity solutions for zeroth,first and second order
        
        self.ur = np.zeros((3,self.__epts,self.__zpts)) #3D matrix containing the axial velocity solutions for zeroth,first and second order
        
        self.div = np.zeros((3,self.__epts,self.__zpts)) #3D matrix containing the plume divergence solutions for zeroth,first and second order
        
        self.lnn = np.zeros((3,self.__epts,self.__zpts)) #3D matrix containing the natural logarothm of density solutions for zeroth,first and second order
        
        self.T = np.zeros((3,self.__epts,self.__zpts)) #3D matrix containing the plasma temperature solutions for zeroth,first and second order
        
        self.phi = np.zeros((3,self.__epts,self.__zpts)) #3D matrix containing the electric potential solutions for zeroth,first and second order
        
        self.eta_ = np.zeros((3,self.__epts,self.__zpts)) #3D matrix containing the eta lines for zeroth,first and second order
                
        """COMPUTE COLD PLASMA BEAM (ZEROTH ORDER) SOLUTION"""
            
        self.uz_0 = np.zeros((self.__epts, self.__zpts)) #matrix containing cold plasma axial velocity
        self.ur_0 = np.zeros((self.__epts, self.__zpts)) #matrix containing zeroth order radial velocity
        n_0  = np.zeros((self.__epts, self.__zpts)) #matrix containing zeroth order plasma density
        self.lnn_0  = np.zeros((self.__epts, self.__zpts)) #matrix containing natural logarithm of density
        
        """ADVANCES IN AXIAL DIRECTION"""
        
        for i in range(0,self.__zpts):
            
            "Calculation of properties at plume axis r = 0"
            
            z = self.z_grid[0,i] # Step in z-grid
            
            n_0[0,i] = self.n0[0] / (1 + self.d0p[0] * z)**2 #COMPUTATION OF DENSITY WHEN r0 --> 0, ON THE AXIS.Apply L'Hopital to equation 3.2 in paper"
            
            self.uz_0[0,i] = self.uz0[0] #Axial velocity at axis is kept constant and equal to origin velocity
                
            self.ur_0[0,i] = 0 #No radial velocity at axis
            
            for j in range(1,self.__epts):
                
                "ADVANCE IN THE RADIAL DIRECTION"
                
                r0 = self.eta[j] #Updating radial coordinate
                
                self.uz_0[j,i] = self.uz0[j] #Updating axial velocity
                
                self.ur_0[j,i] = self.uz0[j]*self.d0[j]  #updating radial velocity
                
                "COMPUTATION OF DENSITY"
                    
                n_0[j,i] = self.n0[j] / ((1 + self.d0p[j] * z) * (1 + self.d0[j] * z / r0 )) #updating density          
                   
        "Updating plume variable arrays with Zeroth-Order plasma plume reults"
        
        self.lnn_0[:,:] = np.log(n_0[:,:])
        
        self.uz[0,:,:]  = self.uz_0
        
        self.ur[0,:,:]   = self.ur_0
        
        self.lnn[0,:,:] = self.lnn_0
        
        self.div[0,:,:] = self.ur[0,:,:]/ self.uz[0,:,:]
        
        self.T[0,:,:]  = super(AEM,self).temp(n_0[:,:],self.n0[0],self.T_0,self.Gamma) #Calling parent class method Hyperplume.temp() to calculate plume cold beam temperature based on density

        self.phi[0,:,:] = super(AEM,self).phi(n_0[:,:],self.n0[0],self.T_0,self.Gamma,self.q_ion) #Calling parent class method Hyperplume.phi() to calculate plume cold beam potential based on density
        
        self.eta_[0,:,:] = self.r_grid-self.div[0,:,:]*self.z_grid #Zeroth order eta-lines based on theory (paper equation no.)
        
        """COMPUTE FIRST ORDER CORRECTION OF PLUME VARIABLES"""
            
        zed = np.zeros((self.__zpts,1)) #array of axial steps
        
        zed[:,0] = self.z_grid[0,:] #redimensioning
            
        zstep = zed[1]-zed[0] #integer measuing the axial step of the grid.Assumes linearly spaced z points
        
        dlnn_dz_0, dlnn_dr_0 = self.partial_derivs(self.lnn_0, 0) #GET REQUIRED DERIVATIVES OF ZEROTH ORDER DENSITY (LOGARITHMIC) ALONG GRID POINTS
            
        "COMPUTE FIRST ORDER CONTRIBUTION OF VELOCITIES. EULER METHOD APPLIED TO ALL STREAMLINES"
        
        self.uz_1    = np.zeros((self.__epts,self.__zpts)) #matrix containing first order axial velocity
        self.ur_1    = np.zeros((self.__epts,self.__zpts)) #matrix containing first order radial velocity
        n1 = np.zeros((self.__epts,self.__zpts)) #matrix containing first order radial velocity
        
        for i in range(1,self.__zpts):
            
            "Computation of the first order velocity perturbation along one streamline. Euler method applied to all the streamlines at the same time"
            
            self.uz_1[:,i] = self.uz_1[:,i-1] + zstep * self.uz0[0]**2 / self.uz0 * (  1/self.uz0[0]**2 * self.uz0p / (1+self.d0p*zed[i-1]) * (self.d0 * self.uz_1[:,i-1] - self.ur_1[:,i-1]) - (n_0[:,i-1]/self.n0[0]) ** (self.Gamma-1) * dlnn_dz_0[:,i-1] )
                   
            self.ur_1[:,i] = self.ur_1[:,i-1] + zstep * self.uz0[0]**2 / self.uz0 * (  1/self.uz0[0]**2 * self.duz0p / (1+self.d0p*zed[i-1]) * (self.d0 * self.uz_1[:,i-1] - self.ur_1[:,i-1]) - (n_0[:,i-1]/self.n0[0]) ** (self.Gamma-1) * dlnn_dr_0[:,i-1] )
        
        "GET REQUIRED VECTORS AND DERIVATIVES TO BE USED IN FISRT ORDER DENSITY CORRECTION"
        
        rur_1 = self.ur_1 * self.r_grid
        
        _,drur1_dr = self.partial_derivs(rur_1,0)
        
        duz1_dz,_ = self.partial_derivs(self.uz_1,0)
            
        """GET FIRST ORDER CONTRIBUTION OF THE DENSITY LOGARITHIM. EULER METHOD APPLIED TO ALL STREAMLINES.tHE FUNCTION INTEGRATES THE
        FIRST ORDER DENSTY PERTURBATION ALONG STREAMLINE"""
        
        self.lnn_1   = np.zeros((self.__epts,self.__zpts)) #matrix containing first order density correction(logarithmic)
        
        "Compute limit of 1/r*drur1/dr for r-->0 as second order derivative of r*ur1.Apply L'Hopital"
        
        limit = np.zeros((1,self.__zpts))  #array containing 1/r(0) * drur1(0)/dr (at the axis of plume)
        _,limit = self.partial_derivs(drur1_dr,0)
        
        for i in range(1,self.__zpts):
            
            "COMPUTE DENSITY FIRST ORDER SOLUTION FOR THE AXIS STREAMLINE"
            
            self.lnn_1[0,i] = self.lnn_1[0,i-1] + zstep * 1 / self.uz0[0] *( - self.uz_1[0,i-1] * dlnn_dz_0[0,i-1] - self.ur_1[0,i-1] * dlnn_dr_0[0,i-1] - duz1_dz[0,i-1] - limit[1,i-1] )
        
        for i in range(1,self.__zpts):
            
            "COMPUTE DENSITY FIRST ORDER SOLUTION FOR ALL REMAINING STREAMLINES"
            
            self.lnn_1[1:,i] = self.lnn_1[1:,i-1] + zstep * 1 / self.uz0[1:] *( - self.uz_1[1:,i-1] * dlnn_dz_0[1:,i-1] - self.ur_1[1:,i-1] * dlnn_dr_0[1:,i-1] - duz1_dz[1:,i-1] - 1 / self.r_grid[1:,i-1] * drur1_dr[1:,i-1])
        
        """Updating plume variable arrays with First Order Correction plasma plume reults"""
        
        self.uz[1,:,:]  = self.uz[0,:,:] + self.uz_1*self.eps
        
        self.ur[1,:,:]   = self.ur[0,:,:] + self.ur_1*self.eps
        
        self.lnn[1,:,:] = self.lnn[0,:,:] + self.lnn_1*self.eps
        
        n1[:,:] = np.exp(self.lnn[1,:,:])
        
        self.div[1,:,:] = self.ur[1,:,:]/ self.uz[1,:,:]
        
        self.T [1,:,:]  = super(AEM,self).temp(n1[:,:],self.n0[0],self.T_0,self.Gamma) #Calling parent class method Hyperplume.temp() to calculate plume second order solution temperature based on density

        self.phi[1,:,:] = super(AEM,self).phi(n1[:,:],self.n0[0],self.T_0,self.Gamma,self.q_ion) #Calling parent class method Hyperplume.phi() to calculate plume second order solution temperature based on density
        
        self.eta_[1,:,:] = self.r_grid-self.div[1,:,:]*self.z_grid #PABLO20170426this is clearly wrong. But why??, corrections are integrable along zeroth order streamlines as said in article.Matlab code calls function orbit2d.m , which I cannot understand
        
        """COMPUTE SECOND ORDER SOLUTION"""
        
        dlnn_dz_0, dlnn_dr_0 = self.partial_derivs(self.lnn_0, 0) #COMPUTE REQUIRED VECTORS AND DERIVATIVES TO BE USED IN SECOND ORDER CORRECTION CALCULATIONS
        
        dlnn_dz_1, dlnn_dr_1 = self.partial_derivs(self.lnn_1, 0)
        
        duz_dz_1, duz_dr_1 = self.partial_derivs(self.uz_1, 0)
        
        dur_dz_1, dur_dr_1 = self.partial_derivs(self.ur_1, -1) 
        
        self.uz_2    = np.zeros((self.__epts,self.__zpts)) #matrix containing second order axial velocity
        self.ur_2    = np.zeros((self.__epts,self.__zpts)) #matrix containing second order radial velocity
        n2 = np.zeros((self.__epts,self.__zpts)) #matrix containing second order radial velocity
        
        for i in range(1,self.__zpts):
            
            "Computation of the second order velocity perturbation along one streamline. Euler method applied to all the streamlines at the same time"
            
            self.uz_2[:,i] = self.uz_2[:,i-1] + zstep * 1 / self.uz0 *( self.uz0p / ( 1 + self.d0p*zed[i-1] )*( self.d0* self.uz_2[:,i-1] - self.ur_2[:,i-1] )-self.uz_1[:,i-1] * duz_dz_1[:,i-1] - self.ur_1[:,i-1] * duz_dr_1[:,i-1]- self.uz0[0]**2*((n_0[:,i-1]/self.n0[0])**(self.Gamma-1) * ( (self.Gamma-1)*self.lnn_1[:,i-1] * dlnn_dz_0[:,i-1] + dlnn_dz_1[:,i-1] ) ))
                   
            self.ur_2[:,i] = self.ur_2[:,i-1] + zstep * 1 / self.uz0 * ( self.duz0p / ( 1 + self.d0p*zed[i-1] )*( self.d0 * self.uz_2[:,i-1] - self.ur_2[:,i-1] )-self.uz_1[:,i-1] * dur_dz_1[:,i-1] - self.ur_1[:,i-1] * dur_dr_1[:,i-1]- self.uz0[0]**2*((n_0[:,i-1]/self.n0[0])**(self.Gamma-1) * ( (self.Gamma-1)*self.lnn_1[:,i-1] * dlnn_dr_0[:,i-1] + dlnn_dr_1[:,i-1] ) ))
        
        "GET REQUIRED VECTORS AND DERIVATIVES TO BE USED IN FISRT ORDER DENSITY CORRECTION"
        
        rur_2 = self.ur_2 * self.r_grid
        
        _,drur2_dr = self.partial_derivs(rur_2,0)
        
        duz2_dz,_ = self.partial_derivs(self.uz_2,0)
            
        """GET SECOND ORDER CONTRIBUTION OF THE DENSITY LOGARITHIM. EULER METHOD APPLIED TO ALL STREAMLINES.tHE FUNCTION INTEGRATES THE
        FIRST ORDER DENSTY PERTURBATION ALONG STREAMLINE"""
        
        self.lnn_2 = np.zeros((self.__epts,self.__zpts)) #matrix containing first order density (logarithmic)
        
        
        "Compute limit of 1/r*drur2/dr for r-->0 as second order derivative of r*ur2.Apply L'Hopital"
        
        limit = np.zeros((1,self.__zpts)) #array containing 1/r(0) * drur1(0)/dr (at the axis of plume)
        _,limit = self.partial_derivs(drur2_dr,0)
            
        for i in range(1,self.__zpts):
            
            "COMPUTE SOLUTION FOR THE AXIS STREAMLINE"
            
            self.lnn_2[0,i] = self.lnn_2[0,i-1] + zstep * 1 / self.uz0[0] *( - self.uz_2[0,i-1] * dlnn_dz_0[0,i-1] - self.ur_2[0,i-1] * dlnn_dr_0[0,i-1] - duz2_dz[0,i-1] - limit[0,i-1] - self.uz_1[0,i-1]*dlnn_dz_1[0,i-1] - self.ur_1[0,i-1]*dlnn_dr_1[0,i-1] )
        
        for i in range(1,self.__zpts):
            
            "COMPUTE SOLUTION FOR ALL REMAINING STREAMLINES"
            
            self.lnn_2[1:,i] = self.lnn_2[1:,i-1] + zstep * 1 / self.uz0[1:] * (-self.uz_2[1:,i-1] * dlnn_dz_0[1:,i-1] - self.ur_2[1:,i-1] * dlnn_dr_0[1:,i-1] - duz2_dz[1:,i-1]- 1/self.r_grid[1:,i-1] * drur2_dr[1:,i-1]- self.uz_1[1:,i-1]*dlnn_dz_1[1:,i-1] - self.ur_1[1:,i-1]*dlnn_dr_1[1:,i-1] )
        
        "Updating plume variable arrays with Second-Order plasma plume reults"
        
        self.uz[2,:,:]  = self.uz[1,:,:] + self.uz_2*self.eps**2
        
        self.ur[2,:,:]   = self.ur[1,:,:] + self.ur_2*self.eps**2
        
        self.lnn[2,:,:] = self.lnn[1,:,:] + self.lnn_2*self.eps**2
        
        n2[:,:] = np.exp(self.lnn[2,:,:])
        
        self.div[2,:,:] = self.ur[2,:,:]/ self.uz[2,:,:]
        
        self.T [2,:,:]  = super(AEM,self).temp(n2[:,:],self.n0[0],self.T_0,self.Gamma) #Calling parent class method Hyperplume.temp() to calculate plume second order solution temperature based on density

        self.phi[2,:,:] = super(AEM,self).phi(n2[:,:],self.n0[0],self.T_0,self.Gamma,self.q_ion) #Calling parent class method Hyperplume.phi() to calculate plume second order solution temperature based on density
        
        self.eta_[2,:,:] = self.r_grid-self.div[2,:,:]*self.z_grid #PABLO20170426 This is clearly wrong. But why?? corrections are integrable along zeroth order streamlines as said in article.Matlab code calls function orbit2d.m , which I dont understand. Could you explain?

    def marching_solver(self,nsects):
        
        """Marching schem AEM plume solver.AEM Class method marching_solver solves extends the AEM solution downstream by
        reinitializing the method at each new calculated plasma plume front, (r0_front,z0_front,uz0_front,n0_front)
        preventing excessive error growth in the calculations and widening the convergence region of thr AEM model.
        
        Marching_solver method reinitializes the plume initial parameter, with the values calculated in the previous 
        integration step, as many times as indicated by the user in nsects. It then solves the plume expansion incrementally
        by callling the solver method multiple times.
        
        Args:
            nsects (int): number of axial sections or steps (plume fronts), where solver reinitializes the
                          model and integrates the solution agin.
        Usage:
            >>> PlumeAEM = AEM(Plasma,z_span,eta_0,n0,uz0,ur0,AEM_Order) #Creation of AEM plume
            >>> Plume.marching_solver(nsects=100)
            
        Same Plasma attributes from standard solver can be accessed in Method marching_solver, but in this case the method stores only the 
        ith higher order correction specified by the user at plume creation with the input argument sol_order:
        
            lnn (numpy.ndarray): matrix containing density values (logarithmic) for the selected AEM solution order.
            uz (numpy.ndarray): matrix containing axial velocity values for the selected AEM solution order.
            ur (numpy.ndarray): matrix containing radial velocity values for the selected AEM solution order.
            T (numpy.ndarray): matrix containing plasma Temperature values for the selected AEM solution order.
            phi (numpy.ndarray): matrix containing plasma ambipolar electric field for the selected AEM solution order.
            div (numpy.ndarray): matrix containing plume divergence values for the selected AEM solution order.
            
        To access these properties, for instance:
            >>> PlumeAEM.lnn # density values for AEM ith order solution of plume expansion in the grid
            >>> PlumeAEM.uz # axial velocity for AEM ith order solution
            >>> PlumeAEM.T # Temperature values for AEM ith order solution
        """
            
        z_grid_ = np.zeros((self.eta.size,self.z_span.size)) #2D matrix containing the z grid points in marching mode
        
        r_grid_ = np.zeros((self.eta.size,self.z_span.size)) #2D matrix containing the r grid points in marching mode
        
        lnn_ = np.zeros((self.eta.size,self.z_span.size)) #2D matrix containing the density (natural logarithm) in marching mode
        
        uz_ = np.zeros((self.eta.size,self.z_span.size)) #2D matrix containing the plume axial velocity in marching mode
        
        ur_ = np.zeros((self.eta.size,self.z_span.size)) #2D matrix containing the plume radial velocity in marching mode
        
        div_ = np.zeros((self.eta.size,self.z_span.size)) #2D matrix containing the plume divergence in marching mode
        
        #T_ = np.zeros((self.eta.size,self.z_span.size)) #2D matrix containing theplume temperature in marching mode
        
        #phi_ = np.zeros((self.eta.size,self.z_span.size)) #2D matrix containing the electric potential in marching mode
        
        zsects_val = np.linspace(self.z_span.max()/nsects,self.z_span.max(),nsects) #z_grid points values at the end of each section/front in maching mode
        
        """GENERATION OF FIRST COMPUTATIONAL GRID"""
        
        dz = self.z_span.max()/(self.z_span.size - 1) #minimum integration step in axial direction
        
        zpts = int(round(zsects_val[0]/dz) + 1) #number of steps in axial direction (number od z_points)
        
        zed = np.zeros((1,zpts+1)) #array containing integration axial steps 
        
        zed[0,0:zpts] = np.linspace(0,zsects_val[0],zpts)
        
        zed[0,zpts] = zsects_val[0] + dz
        
        self.z_grid,self.r_grid = self.grid_setup(zpts+1,self.eta.size) #calculation of z_grid and r_grid based on first front
        
        M0_start = self.M0 #Initial Mach Number based on first front
        
        """Tracking variables for marching method"""
        
        zpts_old = 1 #tracker of previous axial grid points
        count = 0 #counter of axial sections
        vel_factor = 1 #factor for front plume velocity correction
        Te_factor  = 1 #factor for front plume temperature corrections
        
        
        for i in range(nsects):
            
            """Solving current plasma axial extension"""
            
            self.solver() #Calling to method AEM class method to solve the plume in the entire grid
            
            if i == 0:
                
                """Update contents of marching plume_final result matrixes with 
                results from self.solver method in the grid, for first axial section"""
                
                lnn_[:,0:zpts] = self.lnn[self.order,:,0:zpts]
            
                uz_[:,0:zpts] = self.uz[self.order,:,0:zpts]
                
                ur_[:,0:zpts] = self.ur[self.order,:,0:zpts]
                
                div_[:,0:zpts] = self.div[self.order,:,0:zpts]
                
                #T_[:,0:zpts] = self.T[self.order,:,0:zpts]
                
                #phi_[:,0:zpts] = self.phi[self.order,:,0:zpts]
                
                z_grid_[:,0:zpts] = self.z_grid[:,0:zpts]
                
                r_grid_[:,0:zpts] = self.r_grid[:,0:zpts]
                
            else:
                
                """Update contents of marching plume_final result matrixes with 
                results from self.solver method in the grid, for remaining axial section"""
                
                lnn_[:,zpts_old-1:zpts_old+zpts-1] = self.lnn[self.order,:,0:zpts]
            
                uz_[:,zpts_old-1:zpts_old+zpts-1] = self.uz[self.order,:,0:zpts] * vel_factor
                
                ur_[:,zpts_old-1:zpts_old+zpts-1] = self.ur[self.order,:,0:zpts] * vel_factor
                
                div_[:,zpts_old-1:zpts_old+zpts-1] = self.div[self.order,:,0:zpts]
                
                #T_[:,zpts_old-1:zpts_old+zpts-1] = self.T[self.order,:,0:zpts] * Te_factor
                
                #phi_[:,zpts_old-1:zpts_old+zpts-1] = self.phi[self.order,:,0:zpts]
                
                z_grid_[:,zpts_old-1:zpts_old+zpts-1] = self.z_grid[:,0:zpts] + zsects_val[i-1]
                
                r_grid_[:,zpts_old-1:zpts_old+zpts-1] = self.r_grid[:,0:zpts]
                
            zpts_old = zpts_old + zpts -1 #Updating value of axial points, to move forward along the sections
            
            """PREPARING NEXT INTEGATION AXIAL STEP"""
            
            if i < nsects-1:
                
                """Updating new integration interval parameters"""
                
                zpts = int(round((zsects_val[i+1]-zsects_val[i])/dz) + 1) #Updating next number of axial points
        
                zed = np.zeros((1,zpts+1)) #Updating next array of axial integration steps
        
                zed[0,0:zpts] = np.linspace(zsects_val[i],zsects_val[i+1],zpts) - zsects_val[i]
        
                zed[0,zpts] = zed[0,zpts-1] + dz
                
                """New initial profiles for the next axial integration interval"""
                
                r0_front = self.r_grid[:,-2] #New initial r_span obtained from before-last previous section r_grid 
                z0_front = zed #New z0_front based on zed array 
                
                n0_front = np.exp(self.lnn[self.order,:,-2]) #New initial density profile obatained from before-last previous section density
                
                uz0_front = self.uz[self.order,:,-2] #New initial axial velocity profile obatained from before-last previous section axial velocity
                
                ur0_front = self.ur[self.order,:,-2] #New radial velocity profile obatained from before-last previous section radial velocity

                super(AEM,self) .__init__(self.plasma,z0_front.reshape(zed.shape[1],),r0_front,n0_front) #calling Hyperplume constructor to reload the new initial profiles
                
                self.uz0,self.ur0,self.d0 = uz0_front,ur0_front,ur0_front/uz0_front #reloading remaining new plume initial profiles
                
                """Compute derivatives of new initial plume profiles"""
                
                self.d0p = self.eta_deriver(self.eta,self.d0) #derivative of plume divergence
                self.d0p[0],self.d0p[-1] = self.d0[1]/self.eta[1],self.d0p[-2] + (self.d0p[-2] - self.d0p[-3])
                self.uz0p = self.eta_deriver(self.eta,self.uz0) #derivative of initial axial velocity
                self.uz0p[0],self.uz0p[-1] = 0,self.uz0p[-2] + (self.uz0p[-2] - self.uz0p[-3])
                self.duz0p = self.eta_deriver(self.eta,self.d0*self.uz0) #derivatie of initial radial velocity
                self.duz0p[0],self.duz0p[-1] = self.duz0p[1]/self.eta[1],self.duz0p[-2] + (self.duz0p[-2] - self.duz0p[-3])
                
                """Updating correction factors"""
                
                vel_factor = vel_factor * self.uz[self.order,0,-2]/self.uz[self.order,0,0] #velocity factor for next integration interval
      
                Te_factor = Te_factor * self.T[self.order,0,-2]/self.T[self.order,0,0]  #Temperature factor for next integration interval
                
                self.M0 = M0_start*np.sqrt(vel_factor**2/Te_factor) #Update new Mach number based on previos Mach and correction factors
                
                self.eps = 1/self.M0**2 #new AEM epsilon expansion parameter
                
                self.z_grid,self.r_grid = self.grid_setup(zpts+1,self.eta.size) #new interval solution grids
             
                """line1 = plt.plot(self.eta,self.n0,'b');
                
                line2 = plt.plot(self.eta,self.uz0/self.uz0[0],'k')
                
                line3 = plt.plot(self.eta,self.d0,'r',label=r'$Plume divergence \delta = u_{r0}/u_{z0}$')
                
                count = count + 1
                
                if count > 100:
                    plt.close(fig)
                    count = 0
        
        red_line = mlines.Line2D([], [], color='red', label=r'$\delta_{0}$')
        
        black_line = mlines.Line2D([], [], color='black', label=r'$\tilde{u}_{z0}$')
        
        blue_line = mlines.Line2D([], [], color='blue', label=r'$\tilde{n}_{0}$')
        
        plt.legend(handles=[red_line,black_line,blue_line],loc='best')
        
        plt.title('AEM Marching Solver plume front evolution')
        
        plt.xlabel(r'$\eta$')
        
        plt.savefig('Marching_solver_init_fronts.png')"""
                    
        """ Updating final plume attibutes structure with marching_plume results"""
        
        self.z_grid = z_grid_
        
        self.r_grid = r_grid_
        
        self.lnn = lnn_
        
        self.uz = uz_
        
        self.ur = ur_
        
        self.div = div_
        
        self.T = super(AEM,self).temp(np.exp(self.lnn),np.exp(self.lnn)[0,0],self.T_0,self.Gamma)#T_
        
        self.phi = super(AEM,self).phi(np.exp(self.lnn),np.exp(self.lnn)[0,0],self.T_0,self.Gamma,self.q_ion)#phi_
        
        
    def query(self,z,r):
        
        """ Method query returns the density, velocity profile, temperature, the electric potential at
        particular (z,r) points in the Plume.
        
        These plasma properties are interpolated along the previously calculated 2D grids z_grid and r_grid 
        at targeted (z,r) points specified by the user. User must always check if np.max(r) > np.max(self.r_grid),
        np.max(z) > np.max(self.z_grid) in their query point set,to avoid extrapolation results.
        
        Args:
            z (float,numpy.ndarray): new interpolation z points.
            r (float,numpy.ndarray): new interpolation r points.
        
        Outputs:
            lnn (int,numpy.ndarray): logarithmic plasma density at specified (z,r) points in plume grid
            u_z (int,numpy.ndarray): plasma axial velocity at specified (z,r) points in plume grid
            u_r (int,numpy.ndarray): plasma radial velocity at specified (z,r) points in plume grid
            T (int,numpy.ndarray): plasma temperature at specified (z,r) points in plume grid
            phi (int,numpy.ndarray): plasma ambipolar electric potential at specified (z,r) points in plume grid
            eta (int,numpy.ndarray): ion current stream lines at specified (z,r) points in plume grid
            
        Usage:
            >>> z,r = np.linspace(0,100,50),np.linspace(0,50,40) #target (z,r) for plume query
            >>> lnn,u_z,u_r,T,phi,eta=PlumeAEM.query(z,r)
        
        Method query returns only the self.order solution indicated by the user.
        
        #PABLO20170506: After reading extensively on how to perform interpolation over 2D rectagunlar grids,
        I decided to leave method griddata for such task. The problem with interp2D is that given the 
        great size of self.z_grid,and r_grid (mxn), the number of points exceeds memory of method and return error.
        Even if the size of the grids is made smaller (losing accuracy and information in the AEM) the method
        interp2D takes a lot of time (sometimes I had to reset the console). If you want to try to fix the bug or
        otherwise tell what I am doing incorrectly I leave the line of code with interp2D that I was using to solve
        the interpolation (The syntax is exactly the same as the one we saw the ither day in your office, but for
        some reason it is not returning the results i expect.
        
        lnn = interp2d(self.z_grid,self.r_grid,self.lnn[self.order,:,:])(z.flatten(),r.flatten())
        
        On the other hand, griddadta in python does not behave like the Matlab function(In Matlab this function does indeed
        extrapolate the results).Griddata is the recommended function to use over large arrays of data over 2D structured
        or unstructured data, and I have check the return of the interpolation using griddata and it is correct
        """
        
        grid_points = np.array((self.z_grid.flatten(),self.r_grid.flatten())).T #pairing each z_grid point to its matching r_grid point
        
        lnn = griddata(grid_points,self.lnn[self.order,:,:].flatten(),(z,r),method='linear') #Logarithm of plasma plume density interpolation matrix
        
        u_z = griddata(grid_points,self.uz[self.order,:,:].flatten(),(z,r),method='linear') #Plasma plume axial velocity interpolation matrix
        
        u_r = griddata(grid_points,self.ur[self.order,:,:].flatten(),(z,r),method='linear') #Plasma plume radial velocity interpolation matrix
        
        T = griddata(grid_points,self.T[self.order,:,:].flatten(),(z,r),method='linear') #Plasma plume temperature  interpolation matrix
        
        phi = griddata(grid_points,self.phi[self.order,:,:].flatten(),(z,r),method='linear') #Plasma plume electric potential interpolation matrix
        
        eta = griddata(grid_points,self.eta_[self.order,:,:].flatten(),(z,r),method='linear') #Eta-line interpolated values at specied (z,r) points
        
        return lnn,u_z,u_r,T,phi,eta
            
    def grid_setup(self,zpts,epts):
        
        """ grid_setup creates an strctured grid of z,r points where the AEM problem will be integrated
        
        Args:
            zpts (int): number of axial points in the structure. Indicates legnth oof axial plume span
            epts (int): number of radial points in the structure. Indicates legnth of radial plume span
            
        Returns:
            z_grid (numpy.ndarray): 2D matrix containing axial grid points for model integration
            r_grid (numpy.ndarray): 2D matrix containing radial grid points for model integration
            
        Usage:
            >>> z_grid,r_grid = PlumeAEM.grid_setup(100,50)
        """
        z_grid = np.zeros((epts,zpts)) #2D Matrix of Plume z points
        
        r_grid = np.zeros((epts,zpts)) #2D Matrix of Plume r points
        
        """Compute the radial and axial coordinates of each grid points along the streamlines"""
        
        for j in range(epts): #advance radially
            
            for k in range(zpts): #advance axially
                
                "Compute the axial coordinate along the jth streamline"
    
                z_grid[j,k] = self.z_span[k] #update z_grid point from initial z_front
        
        for j in range(epts): #advance radially
            
            r_grid[j,0] = self.eta[j] #updating initial r_grid fron points
            
            for k in range(1,zpts): #advance axially
                
                r_grid[j,k] = r_grid[j,0]+self.d0[j]*self.z_span[k] #update r_grid points from initial front and divergence
                
        return z_grid,r_grid
               
               
    def val_domain(self):
        
        """ val_domain class method evaluates the validity of the AEM series expansion  solution
         at z_grid and r_grid points in the plume. Validity results for each AEM order
        are stored  in the 3D matrix Plume.val. These matrix is filled with values indicating a specific validity condition
        in the results.
        
        VALIDITY VALUES
            0 - Not valid for both velocity and density
            1 - Valid only for velocity
            -1 - Valid only for density
            2 - Valid for both velocity and density
            
        Usage:
            >>> PlumeAEM.val_domain() #Intialize validity condition study
            >>> print(Plume.val) #See results of validation 
        
        """
        
        rel_size = 0.1 #Maximum relative size of the ith order perturbation wrt(i-1)th order to ensure validity of the solution
        
        self.val = np.zeros((3,self.__epts,self.__zpts)) #3D matrix with validity values for each grid point and each solution order
        
        self.val[0,:,:]=2 #Setting valitidity od Zeroth Order solution as reference
            
        for i in range(self.order):
                
                "Creation of plume variable contribution matrixes"
                
                uz_contribution= np.zeros((self.__epts,self.__zpts))
                ur_contribution= np.zeros((self.__epts,self.__zpts))
                lnn_contribution= np.zeros((self.__epts,self.__zpts))
                
                uz_contribution[:,1:] = abs((self.uz[i+1,:,1:]-self.uz[i,:,1:])/self.uz[i,:,1:])
                ur_contribution[1:,1:] = abs((self.ur[i+1,1:,1:]-self.ur[i,1:,1:])/self.ur[i,1:,1:])
                lnn_contribution[:,1:] = abs((self.lnn[i+1,:,1:]-self.lnn[i,:,1:])/self.lnn[i,:,1:])
                
                for j in range(self.__epts): #advance in radial direction
                    
                    for k in range(self.__zpts): #advance axially
                        
                        if (uz_contribution[j,k]<rel_size and ur_contribution[j,k] < rel_size): #applying validity criterion stated in dissertation
                            
                            self.val[i+1,j,k] = 1
                            
                            if lnn_contribution[j,k] < rel_size: #applying validity criterion stated in dissertation
                            
                                
                                self.val[i+1,j,k] = 2
                                
                        elif lnn_contribution[j,k] < rel_size: #applying validity criterion stated in dissertation
                            
                            self.val[i+1,j,k] = -1
                            
                 
    def partial_derivs(self,var,type2):
        
        """Class method partial_derivs computes the partial derivatives of plasma variables with respect to 
        the physical z,r at the plume grid points.
        
        Args:
            var (numpy.ndarray): Variable values to derive at z,r grid points
        
        type2 (int): Integer defining the behaviour of the derivative at the borders and therefore the
                     Type of varible to be differentiated:
        
                   0: Symmetric function. Border derivative value is set to 0
                   
                   -1; Anti-symmetric function. Forward finite difference is used for border derivative calculation.
                   
       Returns:
           dvar_dz  : z-partial derivative values of input argument var at the grid points
           dvar_dr  : r-partial derivative values of input argument var at the grid points
           
       Usage:
           >>> dlnn0_dz,dlnn0dr = PlumeAEM.partial_derivs(Plume.lnn_0) #derivative of Zeroth order density correction
       
       """
    
        dvar_dz = np.zeros((self.__epts,self.__zpts)) #2D matrix storing z-derivative values of var at grid points
        
        dvar_dr = np.zeros((self.__epts,self.__zpts)) #2D matrix storing z-derivative values of var at grid points
        
        zfactor = np.zeros((self.__epts,self.__zpts)) #2D Jacobian z tranformation matrix to pass from zita-eta derivatives to z,r derivatives at grid points(see dissertation appendix) 
        
        rfactor = np.zeros((self.__epts,self.__zpts)) #2D Jacobian r tranformation matrix to pass from zita-eta derivatives to z,r derivatives at grid points(see dissertation appendix) 
        
        eta_grid = np.dstack((self.r_grid[:,0],)*self.__zpts)[0] #2D matrix containing stacked values of the eta coordinates at grid points
        
        "CALCULATION OF DERIVATIVES IN ZITA-ETA COORDINATES AT GRID POINTS" 
            
        "This function computes the partial derivatives with respect to the stream coordinates eta-Zita at the grid points"
        
        dvar_dzita = np.zeros((self.__epts,self.__zpts)) #2D matrix storing zita-derivative values of var at grid points
        
        dvar_deta = np.zeros((self.__epts,self.__zpts)) #2D matrix storing eta-derivative values of var at grid points
        
        "Compute the zita partial derivatives of the grid points inside the domain (excluding the boundaries) with a centred difference"
        
        dvar_dzita[0:,1:-1] = (var[0:,2:] - var[0:,0:-2]) / (self.z_grid[0:,2:] - self.z_grid[0:,0:-2])
        
        "Compute the zita partial derivatives at z = 0 and zmax, using respectively a forwards and backwards Euler derivative"
        
        dvar_dzita[0:,0] = ( var[0:,1] - var[0:,0] ) / (self.z_grid[0,1] - self.z_grid[0,0])
        
        dvar_dzita[0:,-1] = ( var[0:, -1] - var[0:, -2] ) / (self.z_grid[0,-1] - self.z_grid[0,-2])
        
        """Compute the eta partial derivatives of the grid points inside the domain (excluding the boundaries) with a centred difference even when points
            are not uniform along eta"""
        
        h1 = eta_grid[1:-1,0:] - eta_grid[0:-2,0:] #steps in eta-grid used for derivation of variable 
        
        h2 = eta_grid[2:,0:] - eta_grid[1:-1,0:]
        
        dvar_deta[1:-1,0:] = ((h1**2) * var[2:,0:] - ((h1**2) - (h2**2)) * var[1:-1,0:] - (h2**2) * var[0:-2,0:] ) / ((h1**2)*h2 + h1*(h2**2))
        
        """ Compute the eta partial derivatives at z = 0 and zmax, using respectively a forwards and backwards Euler derivative
            For eta = 0, consider the type of input function"""
            
        if type2 == 0:
            
            dvar_deta[0,0:] = 0
            
        else:
            
            dvar_deta[0,0:] = ( var[1, 0:] - var[0, 0:] ) / (eta_grid[1, 0:] - eta_grid[0,0:]) 
            
        "Take the final eta line derivative equal to that of the previous point"
            
        dvar_deta[-1,0:] = dvar_deta[-2,0:] + (dvar_deta[-2,0:] - dvar_deta[-3,0:])
            
        for i in range(0,self.__zpts):
            
            "Compute transformation factors in matrix form for z and r derivative using the Jacobian matrix"
            
            zfactor[:,i] = -self.d0[:] / (1 + self.z_grid[:,i] * self.d0p[:]) #Updating Jacobian matrixes
            
            rfactor[:,i] = 1 / (1 + self.z_grid[:,i] * self.d0p[:])
        
        dvar_dz[:,:] = dvar_dzita + zfactor * dvar_deta #updatinf final derivatives in z,r coordinates
        
        dvar_dr[:,:] = rfactor * dvar_deta
        
        return dvar_dz, dvar_dr
        
