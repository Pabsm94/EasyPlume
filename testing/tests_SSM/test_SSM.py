# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 17:18:54 2016

@author: pablo
"""
import os 

dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
 
sys.path.append(dir_path) #change to src

from src import np,unittest,SSM,type_parks,type_korsun,type_ashkenazy,Hyperplume

class Test_SSM_plume(unittest.TestCase):
    
    """Class Test_SSM_Plume performs different validity tests on Class SSM, by checking on each of its indpendent methods"""
    
    def test_SSM__init(self):
        
        """ Testin SSM interface.Several calling methods for class SSM."""
        
        Plume1 = SSM() #Creation of SSM plume with default input arguments
        
        self.assertIsInstance(Plume1,SSM) #checking categorization of Plume1
        
        self.assertIsNotNone(Plume1) #checking object creation of class SSM
        
        self.assertEqual(Plume1.M_0,40) #checking creation of class attributes
        
        self.assertEqual(Plume1.d_0,0.2)
        
        Plasma = Hyperplume().simple_plasma(1,1,1,1) 
        
        Z_span = np.linspace(0,100,500)
    
        eta_0 = np.linspace(0,40,500)
        
        nu = np.exp(- 0.047 * eta_0** 2)
        
        Plume3 = SSM(Plasma,40,0.2,Z_span,eta_0,nu) #alternative creation of SSM plume object with user given inputs
        
        self.assertIsInstance(Plume3,SSM)
        
        self.assertIsNotNone(Plume3)
        
        upsilon = np.ones(nu.size) 
        
        self.assertRaises(TypeError,Plume4 = SSM, args = (Plasma,40,0.2,Z_span,eta_0,nu,upsilon)) # only initial density vector can be passed as input to SSM class. Error should be raised when both initial velocity and density profiles are given
    
    def test_solver(self):
        
        """Tests on SSM Class self.solver() method"""
        
        Plasma = Hyperplume().simple_plasma(1,1,1,1) 
        
        Z_span = np.linspace(0,100,500)
    
        eta_0 = np.linspace(0,40,500)
        
        C_user = 6.15
        
        n0 = np.exp(- C_user/2 * eta_0** 2)
        
        Plume = SSM(Plasma,40,0.2,Z_span,eta_0,n0) #creation of SSM plume object
        
        Plume.solver() #good call on SSM method self.solver
        
        z,r = np.linspace(0,10,5000),np.linspace(0,3,5000)
        
        self.assertRaises(TypeError, Plume.solver, args = (z,r)) #wrong call on self.solver method leads to exception error
        
        self.assertIsNotNone(Plume.nu_prime_interp) #checking performance of self.solver method in storing plume variables
        
        self.assertIsNotNone(Plume.h_interp) #checking trakcking of self-similar dilation function h and dh
        
        self.assertAlmostEqual(Plume.C,C_user,places = 0) #Testing model-calculated dimensioning constant C with user-given constant C up to three decimal places
        
        
    def test_upsilon_compare(self):
        
        """ Comparison between SSM general framework developed in Python code, and theoretical plume profiles"""
        
        P = Hyperplume().simple_plasma(1,1,1,1)
        
        Z_span = np.linspace(0,100,500)
        
        eta_0 = np.linspace(0,40,500)
        
        n0_parks = np.exp(-np.log(0.05)*eta_0**2) #Initial density profile for a Parks-type SSM plume
        
        upsilon_parks = np.ones(eta_0.size) #Initial dimensionless axial velocity  profile for a Parks-type SSM plume
        
        Plume = SSM(P,40,0.2,Z_span,eta_0,n0_parks)
        
        Plume.solver()
        
        self.assertAlmostEqual(float(Plume.upsilon_interp(3)),float(np.any(upsilon_parks)),places=0) #comparing model returned upsilon profile, with theoretical upsilon profile
        
        
    def test_query(self):
        
        """Assertiveness of method query inside SSM plume class"""
        
        P = Hyperplume().simple_plasma(1.6e-19,2,1.5,1.4)
        
        Z_span = np.linspace(0,100,50)
        
        eta_0 = np.linspace(0,40,50)
        
        n0 =  np.exp(-0.05*eta_0**2)
        
        Plume = SSM(P,20,0.2,Z_span,eta_0,n0)
        
        Plume.solver()
        
        z_target = np.array([15,20,25,30])
        
        r_target = np.array([20,25,30,35])
        
        Z_target,R_target = np.meshgrid(z_target,r_target)
        
        n,u_z,u_r,T,phi,error,etaSSM = Plume.query(Z_target,R_target) #calling method query
        
        self.assertIsNotNone(n) #checking performance of self.query method based on returned varables
        
        self.assertIsNotNone(T)
        
        self.assertEqual(n.shape,Z_target.shape,R_target.shape) #checking performance of self.method based on targeted poins inputted by user
        
    def test_types(self):
        
        """ Checking theoretical Parks,Ashkenazy and Korsun model plume creation"""
        
        P = Hyperplume().simple_plasma(1,1,1,1)
        
        Z_span = np.linspace(0,100,500)
        
        eta_0 = np.linspace(0,40,500)
        
        Plume = type_parks()
        
        Plume1 = type_parks(P,30,0.3,Z_span,eta_0,0.5)
        
        self.assertIsNotNone(Plume,Plume1) # test type_ interface.
        
        P1 = Hyperplume().simple_plasma(1,1,1,1.4)
        
        Plume2 = type_parks(P1,30,0.3,Z_span,eta_0,0.5)
        
        Plume3 = type_korsun(P1,30,0.3,Z_span,eta_0,0.5)
        
        Plume4 = type_ashkenazy(P1,30,0.3,Z_span,eta_0,0.5)
        
        self.assertIsNone(Plume2,Plume4) # test validity of Gamma value for the different plume types.
        
        self.assertIsNotNone(Plume3)
        
        nu = nu =  1 / (1 + 0.047 * eta_0** 2)
        
        self.assertRaises(TypeError,Plume5 = type_parks, args =(P1,30,0.3,Z_span,eta_0,0.5,nu)) # for type_ plumes, initial density profile is not an input
        
if __name__ == '__main__':  # When, run for testing only
    
    #self-test code
    
    unittest.main()
    
        
        