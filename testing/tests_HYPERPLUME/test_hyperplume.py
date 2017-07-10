# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 17:18:54 2016

@author: pablo
"""

import os 
    
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
 
sys.path.append(dir_path) #path to src

from src import np,unittest,Hyperplume

class Test_Hyperplume(unittest.TestCase):
    
    """Class Test_AEM checks the interface of the Plume Class Hyperplume, by running test on each of its indpendent methods"""
    
    def test_interface(self):
        
        """ tests on user call methods"""
        
        Object = Hyperplume() #call with preloaded input arguments
        
        self.assertIsNotNone(Object) #checking for object creation
        
        self.assertIsInstance(Object,Hyperplume) #checking for Object classification
        
        self.assertIsNotNone(Object.Gamma) #checking for class attributes initalization
        
        Object2 = Hyperplume(plasma={'Electrons': {'Gamma': 1.3,'T_0_electron': 2.1801714e-19,'q_electron': -1.6e-19},'Ions': {'mass_ion': 2.1801714e-25, 'q_ion': 1.6e-19}},z_span=np.linspace(0,10,500),r_span=np.linspace(0,3,500),n_init=0.0472*np.linspace(1,0,500)**2) #cration of Hyperplume object with different arguments
        
        self.assertIsNotNone(Object2) #checking for object creation
        
        self.assertIsInstance(Object2,Hyperplume) #checking for Object classification
        
        self.assertIsNotNone(Object2.plasma) #checking for class attributes initalization
        
    def test_solver(self):
        
        Object = Hyperplume() 
        
        Object.solver() #testing abstract method Hyperplume.solver()
        
    def test_simple_plasma(self):
        
        Plasma = Hyperplume().simple_plasma(charge=1.6e-19,ion_mass=2.1801714e-25,init_plasma_temp=2.1801714e-19,Gamma=1.2) #creation of simple_plasma object with user given input arguments
        
        self.assertIsNotNone(Plasma) #testing for plasma creation
        
        self.assertTrue(Plasma['Ions']['mass_ion']==2.1801714e-25) #testing for method performance
        
        self.assertTrue(Plasma['Electrons']['Gamma']==1.2)
        
        with self.assertRaises(KeyError):
            
            Plasma['Ions']['T_0_electron'] #testing expected errors in simple_plasma wrong calling
            
            Plasma['Ions']['Gamma']
            
            Plasma['Electrons']['mass_ion']
            
            Plasma['Electrons']['q_ion']
            
    def test_deriver(self):
        
        """Testing Easplume.eta_deriver method"""
        
        x = np.linspace(1,10)
        y = x**2
            
        diff_y = Hyperplume().eta_deriver(x,y) 
        
        self.assertIsNotNone(diff_y) #checking creation of diff_y
        
    def test_plasma_methods(self):
        
        """Different tests on Easplume methods self.n,self.phi,self.temp"""
        
        T1 = Hyperplume().temp(0.043,1,2.1801714e-19,1)
        
        T2 = Hyperplume().temp(0.043,1,2.1801714e-19,1.2)
        
        phi1 = Hyperplume().phi(0.043,1,2.1801714e-19,1.2,-1.6e-19)
        
        phi2 = Hyperplume().phi(0.043,1,2.1801714e-19,1,-1.6e-19)
        
        self.assertIsNotNone(T1) #checking good return of method
        
        self.assertIsNotNone(phi1)
        
        self.assertNotEqual(T1,T2) #checking good performance on method based on thermal expansion model
        
        self.assertNotEqual(phi1,phi2) #checking good performance on method based on thermal expansion model
        
    
        
if __name__ == '__main__':  # When, run for testing only
    
    #self-test code
    
    unittest.main()