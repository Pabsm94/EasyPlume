# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:18:06 2016

@author: pablo
"""

import os 

dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
 
sys.path.append(dir_path) #change to src

from src import np,unittest,AEM,Hyperplume

class Test_AEM_plume(unittest.TestCase):
    
    """Class Test_AEM checks the interface of the Plume Class AEM, by running test on each of its indpendent methods"""
    
    def test_interface(self):
        
        """Tests on User AEM call interface""" 
        
        Plume1 = AEM() #call with preloaded input arguments
        
        self.assertIsInstance(Plume1,AEM) #checking for Object classification
        
        self.assertIsNotNone(Plume1.eps) #checking for class attributes initalization
        
        P = Hyperplume().simple_plasma(1.6e-19,2,1.5,1.4)
    
        Z_span = np.linspace(0,100,20)
    
        eta_0 = np.linspace(0,40,20)
    
        n0 =  np.exp(-0.05*eta_0**2)

        Plume = AEM(P,Z_span,eta_0,n0,np.linspace(1,0.1,20),0.6*np.linspace(1,0.1,20)) #call with user given input arguments
        
        self.assertIsInstance(Plume,AEM) #checking for Object classification
        
        self.assertIsNotNone(Plume.Gamma) #checking for class attributes initalization
        
        
    def test_solvers(self):
        
        """Tests on AEM Class self.solver() and self.maching_solver methods"""
        
        Plume = AEM()
        
        Plume.solver()  #testing particular method AEM.solver() based  on Hyperplume.solver() abstract method
        
        self.assertIsNotNone(Plume.uz_0) #testing the creation of Plume AEM zeroth order correction results
        
        self.assertIsNotNone(Plume.uz[0,:,:]) #testing the creation of Plume properties results for cold beam solution plasma
        
        self.assertTrue(Plume.uz_0.shape == Plume.uz[0,:,:].shape) #testing shape matching and good performance of solver method
        
        self.assertTrue(np.all(Plume.uz_0) == np.all(Plume.uz[0,:,:])) #testing perfect element matching and good performance of solver method
        
        self.assertIsNotNone(Plume.uz_1,Plume.uz[1,:,:]) #testing the creation of Plume AEM higher order correction results
        
        self.assertIsNotNone(Plume.uz_2,Plume.uz[2,:,:]) 
        
        self.assertTrue(Plume.uz_1.shape == Plume.uz[1,:,:].shape) #testing shape matching and good performance of solver method for higher orders
        
        self.assertTrue(Plume.uz_2.shape == Plume.uz[2,:,:].shape) 
        
        self.assertFalse(np.all(Plume.uz_1) == np.all(Plume.uz[1,:,:])) #testing element difference and good performance of solver method on higher order AEM corrections
        
        self.assertFalse(np.all(Plume.uz_2) == np.all(Plume.uz[2,:,:]))
        
        Plasma = Hyperplume().simple_plasma(1.6e-19,2.1801714e-25,2.1801714e-19,1)
    
        Z_span = np.linspace(0,100,2001)
        
        eta_0 = np.linspace(0,3,101)
        
        n0 =  np.exp(-6.15/2*eta_0**2)
        
        uz1 = np.linspace(20000,20000,101)
        ur1 = np.linspace(0,1.607695154586736e+04,101)
        
        Plume1 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
        
        Plume1.marching_solver(100) #testing particular method AEM.marching_solver() 
        
        self.assertIsNotNone(Plume1.z_grid) #testing the creation of Plume properties results 
        
        self.assertIsNotNone(Plume1.lnn,)
        
        self.assertNotEqual(Plume.lnn.shape,Plume1.lnn.shape) #testing good performance of marching_solver
        

    def test_query(self):
        
        """Assertiveness of method query inside AEM plume class"""
        
        P = Hyperplume().simple_plasma(1.6e-19,2,1.5,1.4)
    
        Z_span = np.linspace(0,100,20)
        
        eta_0 = np.linspace(0,40,20)
        
        n0 =  np.exp(-0.05*eta_0**2)
    
        Plume = AEM(P,Z_span,eta_0,n0,np.linspace(1,0.1,20),0.6*np.linspace(1,0.1,20))
        
        Plume.solver()
        
        z_target = np.array([15,20,25,30])
        
        r_target = np.array([20,25,30,35])
        
        ZZ,RR = np.meshgrid(z_target,r_target)
        
        lnn,u_z,u_r,T,phi,eta = Plume.query(ZZ,RR)
        
        self.assertIsNotNone(lnn,T) #tsting good return of query method
        
        self.assertTrue(lnn.shape == (4,4))
        
    
        
        
       
if __name__ == '__main__':  # When, run for testing only
    
    #Testing Routine
    
    unittest.main()
    

        
