# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:51:56 2016

@author: pablo
"""

import sys 

import os 

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','easyplume', 'MOC'))

from MOC_PLUME import MOC

import unittest 

import numpy as np

class Test_MOC(unittest.TestCase):
    
    def test_interface(self):
        
        Plume = MOC()
        
        self.assertIsNotNone(Plume)
        
        self.assertIsNotNone(Plume.v)
        
        P = simple_plasma(1.6e-19,2,1.5,1.4)
    
        r_0 = np.linspace(0,40,20)
    
        n0 =  np.exp(-0.05*r_0**2)
        
        Plume1 = MOC(P,r_0,n0,np.linspace(1,0.1,20),0.6*np.linspace(1,0.1,20))
        
        self.assertIsNotNone(Plume1)
        
        self.assertIsNotNone(Plume1.v)
        
    def test_solver(self):
        
        Plume = MOC()
        
        Plume.solver()  
        
        self.assertIsNotNone(Plume.n_fun)
        
        self.assertIsNotNone(Plume.T_fun)
        
    def test_query(self):
        
        Plume = MOC()
        
        z_target = np.array([15,20,25,30])
    
        r_target = np.array([20,25,30,35])
    
        n,u_z,u_r,T,phi = Plume.query(z_target,r_target)
        
        self.assertIsNotNone(n)
        
        self.assertTrue(n.shape == (len(z_target),len(r_target)))
     
    def test_advance_front(self):
        
        Plume = MOC()
     
        Front_old = np.array([[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]])
         
        Front_new = Plume.Advance_Front(Front_old)
         
        self.assertIsNotNone(Front_new)
         
        self.assertEqual(Front_old.shape,Front_new.shape)
        
    def test_Moc_inner(self):
        
        Plume = MOC()
      
        Front_old = np.array([[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]])
       
        p4 = Plume.Moc_inner(1,2,Front_old)
       
        self.assertIsNotNone(p4)
       
        self.assertTrue(p4.shape == np.shape(Front_old[0,:])) 

if __name__ == '__main__':  # When, run for testing only
    
    #self-test code
    
    unittest.main()
        
        