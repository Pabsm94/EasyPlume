# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:19:24 2016

@author: pablo
"""

import sys,os 

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'EASYPLUME'))

from easyplume import *

class MOC(Easyplume):
    
    def __init__(self,plasma=simple_plasma(1,1,1,1),R_span=np.linspace(0,40,500),n_init=np.linspace(1,100,500),uz_init=np.linspace(1,100,500),ur_init=np.linspace(0,100,500)):
        
        self.Z_span=np.zeros(R_span.size)
        
        self.Gamma,self.T_0,self.m_ion,self.q_ion,_,self.r_span,self.n,self.u,self.v,_ = super(MOC,self).__init__(plasma,self.Z_span,R_span,n_init,uz_init,ur_init)
        
        "Compute dimensionless values of Temperature, potential energy, mechanical energy and sound speed for initial front"
        
        self.T_init=simple_species.temp(self,self.n,self.n[0],self.T_0,self.Gamma)
        
        "Initial potential is zero, as indicated in paper"
        
        self.phi_init = np.zeros(len(self.n))
        self.Hi = np.array( 0.5*self.m_ion*(self.u**2+self.v**2))
        self.c_=np.sqrt(self.Gamma*self.T_init/self.m_ion)
        
    def solver(self):
        
        "LOAD INITIAL FRONT" 
        
        Front_init = np.stack((self.Z_span,self.r_span,self.u,self.v,self.phi_init,self.Hi,self.n,self.T_init,self.c_),axis=-1)
        
        Front_no = len(self.Z_span)
        
        Front_old = np.zeros((Front_no,Front_no))
        
        Front_old = Front_init
        
        z = np.zeros((Front_no,Front_no))
        
        r = np.zeros((Front_no,Front_no))
        
        u = np.zeros((Front_no,Front_no))
        
        v = np.zeros((Front_no,Front_no))
        
        phi = np.zeros((Front_no,Front_no))
        
        Hi = np.zeros((Front_no,Front_no))
        
        n = np.zeros((Front_no,Front_no))
        
        T = np.zeros((Front_no,Front_no))
        
        c = np.zeros((Front_no,Front_no))
        
        z[:,0] = Front_old[:,0]
        
        r[:,0] = Front_old[:,1]
        
        u[:,0] = Front_old[:,2]
        
        v[:,0] = Front_old[:,3]
        
        phi[:,0] = Front_old[:,4]
        
        Hi[:,0] = Front_old[:,5]
        
        n[:,0] = Front_old[:,6]
        
        T[:,0] = Front_old[:,7]
        
        c[:,0] = Front_old[:,8]
        
        Front_new = np.zeros((Front_no,9))
        
        for i in range(1,Front_no):
            
            "Call advance_front to calculate new front"
            
            Front_new = self.Advance_Front(Front_old)
            
            "UPDATE NEW FRONT IN LIST"
            
            z[:,i] = Front_new[:,0]
        
            r[:,i] = Front_new[:,1]
            
            u[:,i] = Front_new[:,2]
            
            v[:,i] = Front_new[:,3]
            
            phi[:,i] = Front_new[:,4]
            
            Hi[:,i] = Front_new[:,5]
            
            n[:,i] = Front_new[:,6]
            
            T[:,i] = Front_new[:,7]
            
            c[:,i] = Front_new[:,8]
            
            Front_old = Front_new
            
        self.n_fun = SmoothBivariateSpline(z.flatten(),r.flatten(),n.flatten())
        
        self.u_fun = SmoothBivariateSpline(z.flatten(),r.flatten(),u.flatten())
        
        self.v_fun = SmoothBivariateSpline(z.flatten(),r.flatten(),v.flatten())
        
        self.T_fun = SmoothBivariateSpline(z.flatten(),r.flatten(),T.flatten())
        
        self.phi_fun = SmoothBivariateSpline(z.flatten(),r.flatten(),phi.flatten())
            
    def query(self,z,r):
        
        n = self.n_fun(z,r)
        
        u_z = self.u_fun(z,r)
        
        u_r = self.v_fun(z,r)
        
        T = self.T_fun(z,r)
        
        phi = self.phi_fun(z,r)
        
        return n,u_z,u_r,T,phi
        
        
    def Advance_Front(self,Front_old):
        
        """This function calculates the kernel of the given front (i.e., no new wall points are calculated. 
        The initial front must be complete."""
        
        Front_points,_ = np.shape(Front_old)
        
        Front_sub = np.zeros((np.shape(Front_old)))
        
        Front_new = np.zeros((np.shape(Front_old)))
        
        "SUBFRONT"
        
        for i in range(Front_points-1):
            
            Front_sub[i,:] = self.Moc_inner(i+1,i,Front_old)
            
        Front_points,_ = np.shape(Front_sub)
        
        "CALCULATE INNER" 
        
        for i in range(1,Front_points-1):
            
            Front_new[i,:] = self.Moc_inner(i,i+1,Front_sub)
        
        "CALCULATE AXIS"
        
        Front_sub[0,:] = Front_old[0,:]
        
        Front_new[0,:] = self.Moc_at_axis(1,0,Front_sub)
        
        return Front_new
            
        
    def Moc_inner(self,i1,i2,Front):
        
        "Function to calculate variables at a new inner point using direct MOC (p1 and p2 are given; p3 is found and interpolated)."
        
        q_m = self.q_ion/self.m_ion
        
        p1,p2 = np.zeros(9),np.zeros(9)
        
        p1,p2 = Front[i1,:],Front[i2,:]
        
        "Complete p1, p2 with mech energy"
        
        p1[5] = self.q_ion*p1[4]  + 0.5*self.m_ion*(p1[2]**2+p1[3]**2)
        
        p2[5] = self.q_ion*p2[4]  + 0.5*self.m_ion*(p2[2]**2+p2[3]**2)
        
        "Variables needed to determine p3"
        
        p3_lzd = p2[2]-p1[2]
        
        p3_lrd = p2[3]-p1[3]
        
        p3_zd = p2[0]-p1[0] 
        
        p3_rd = p2[1]-p1[1]
        
        p3_a = p3_lzd*p3_rd-p3_lrd*p3_zd
        
        p4 = np.zeros(9)
        
        max_iter = 100
        
        corrector_tol = 1e-6
        
        Z,R,U,V,PHI = inf,inf,inf,inf,inf
        
        for iteration in range(max_iter):
            
            if iteration == 0:
                
                "Properties for p,m"
                
                pp = p2
                
                pm = p1
                
            else:
                
                pp[0],pp[1],pp[2],pp[3],pp[4] = (p2[0]+p4[0])/2,(p2[1]+p4[1])/2,(p2[2]+p4[2])/2,(p2[3]+p4[3])/2,(p2[4]+p4[4])/2
                
                pp[6]= simple_species.n(self,self.n[0],self.T_0,pp[4],self.Gamma,-self.q_ion)
                
                pp[7] = simple_species.temp(self,pp[6],self.n[0],self.T_0,self.Gamma)
                
                pp[8] = np.sqrt(self.Gamma*pp[7]/self.m_ion)
                
                pm[0],pm[1],pm[2],pm[3],pm[4] = (p1[0]+p4[0])/2,(p1[1]+p4[1])/2,(p1[2]+p4[2])/2,(p1[3]+p4[3])/2,(p1[4]+p4[4])/2
                
                pm[6]= simple_species.n(self,self.n[0],self.T_0,pm[4],self.Gamma,-self.q_ion)
                
                pm[7] = simple_species.temp(self,pm[6],self.n[0],self.T_0,self.Gamma)
                
                pm[8] = np.sqrt(self.Gamma*pm[7]/self.m_ion)
                
            "Slope terms for lines p,m  "
                
            Lp = charline('p',pp[0],pp[1],pp[2],pp[3],pp[8])
            
            Lm = charline('m',pm[0],pm[1],pm[2],pp[3],pm[8])
            
            denom = Lp[0] * Lm[2] - Lp[2] * Lm[0]
            
            "Obtain position of point 4 and line parameters tm, tp"
            
            tm = (Lp[2]*(Lm[1]-Lp[1]) - Lp[0]*(Lm[3]-Lp[3])) / denom
            
            tp = - (Lm[2]*(Lp[1]-Lm[1]) - Lm[0]*(Lp[3]-Lm[3])) / denom
            
            p4[0] = Lm[0]*tm + Lm[1]
            
            p4[1] = Lm[2]*tm + Lm[3]
            
            z41_ = p4[0]-p1[0]
            
            r41_ = p4[1]-p1[1]
            
            "Obtain position of point 3 Variables needed to find p3"
            
            p3_b = p3_lrd*z41_ - p3_lzd*r41_ - p1[3]*p3_zd + p1[2]*p3_rd
            
            p3_c = p1[3]*z41_ - p1[2]*r41_
            
            s3 = -2*p3_c/(p3_b*(1+np.sqrt(1-4*p3_a*p3_c/p3_b**2)))
            
            p3 = np.zeros(9)
            
            "Interpolate for properties at 3 needed for slope terms"
            
            p3[0] = p1[0]*(1-s3) + p2[0]*s3
            
            p3[1] = p1[1]*(1-s3) + p2[1]*s3
            
            p3[2] = p1[2]*(1-s3) + p2[2]*s3
            
            p3[3] = p1[3]*(1-s3) + p2[3]*s3
            
            p3[5] = p1[5]*(1-s3) + p2[5]*s3
            
            p3[4] = p3[5]/self.q_ion - 0.5 * (p3[2]**2 + p3[3]**2)/q_m
            
            if iteration == 0:
                
                "Properties at po"
                
                po = p3
                
            else:
                
                po[2],po[3] = (p3[2] + p4[2])/2,(p3[3] + p4[3])/2
            
            "Solve equations to calculate properties at 4"
            
            fp = 0
            
            if pp[1] != 0:
            
                fp = -pp[3]/pp[1]
                
            fm = - pm[3]/pm[1]
            
            MC = np.array([[pp[3], -pp[2], -q_m*np.sqrt((pp[2]**2+pp[3]**2)/pp[8]**2 - 1)],[pm[3], -pm[2], +q_m*np.sqrt((pm[2]**2+pm[3]**2)/pm[8]**2 - 1)],[po[2],po[3],q_m]])
            
            Gi = np.zeros((3,1))
            
            Gi[0,0] = np.dot(MC[0,:],np.array([[p2[2]],[p2[3]],[p2[4]]]))
            
            Gi[1,0] = np.dot(MC[1,:],np.array([[p1[2]],[p1[3]],[p1[4]]]))
            
            Gi[2,0] = np.dot(MC[2,:],np.array([[p1[2]],[p1[3]],[p1[4]]]))
            
            Fi = np.zeros((3,1))
            
            Fi[0,0] = (Lp[0]*pp[3]-Lp[2]*pp[2])*fp*tp
            
            Fi[1,0] = (Lm[0]*pm[3]-Lm[2]*pm[2])*fm*tm
            
            Fi[2,0] = 0
            
            "Solve for properties at point 4"
            
            sol = np.dot(np.linalg.inv(MC),(Fi+Gi))
            
            p4[2] = sol[0]
            
            p4[3] = sol[1]
            
            "Enforce exact energy equation along char o"
            
            p4[4] = p3[5]/self.q_ion - 0.5 * (p4[2]**2 + p4[3]**3)/q_m
            
            iter_err = abs(p4[0]-Z)+abs(p4[1]-R)+abs(p4[2]-U)+abs(p4[3]-V)+abs(p4[4]-PHI)
            
            if ( iter_err < corrector_tol ):
                
                break
            
            else:
                
                Z = p4[0] 
                R = p4[1] 
                U = p4[2] 
                V = p4[3] 
                PHI = p4[4]
                
        if iteration == max_iter:
            
            print('Convergence was not achieved')
            
        "Additional properties at 4"
            
        p4[5] = self.q_ion*p4[4]  + 0.5*self.m_ion*(p4[2]**2+p4[3]**2)
        
        p4[6] = simple_species.n(self,self.n[0],self.T_0,p4[4],self.Gamma,-self.q_ion)
        
        p4[7]= simple_species.temp(self,p4[6],self.n[0],self.T_0,self.Gamma)
        
        p4[8] = np.sqrt(self.Gamma*p4[7]/self.m_ion)
            
        return p4
        
    def Moc_at_axis(self,i1,i3,Front):
        
        q_m = self.q_ion/self.m_ion
        
        p1,p3 = Front[i1,:],Front[i3,:]
        
        p3[5] = self.q_ion * p3[4] + 0.5 * self.m_ion * (p3[2]**2 + p3[3]**2)
        
        Lo = np.zeros(4)
        
        Lo[0],Lo[1],Lo[2],Lo[3] = 1,p3[0],0,p3[1]
        
        p4 = np.zeros(9)
        
        max_iter = 100
        
        corrector_tol = 1e-6
        
        Z,R,U,V,PHI = inf,inf,inf,inf,inf
        
        for iteration in range(max_iter):
            
            if iteration == 0:
                
                pm = p1
                
            else:
                
                pm[0],pm[1],pm[2],pm[3],pm[4] = (p1[0]+p4[0])/2,(p1[1]+p4[1])/2,(p1[2]+p4[2])/2,(p1[3]+p4[3])/2,(p1[4]+p4[4])/2
        
                pm[6] = simple_species.n(self,pm[4],self.Gamma)
                
                pm[7] = simple_species.temp(self,pm[6],self.T_init,self.Gamma)
                
                pm[8] = np.sqrt(self.Gamma*pm[7]/self.m_ion)
                
            Lm = charline('m',p1[0],p1[1],pm[2],pm[3],pm[8])
            
            if iteration == 0:
                
                po = p3
                
            else:
                
                po[2],po[3] = (p3[2] + p4[2])/2,(p3[2] + p4[2])/2
                
            tm = (Lm[3]-Lm[1]/Lo[0]+Lo[1]/Lo[0])/(Lo[2]*Lm[0]/Lo[0] - Lm[2])
            
            p4[0] = Lm[0]*tm + Lm[1]
            
            p4[1] = Lm[2]*tm + Lm[3]
                
            fm = -pm[3]/pm[1]
            
            MC = np.array([[pm[3], q_m*np.sqrt(pm[2]**2+pm[3]**2-pm[8]**2)/pm[8]],[po[2],q_m]])
            
            Gi,Fi = np.zeros((2,1)),np.zeros((2,1))
            
            Gi[0,0] = np.dot(MC[0,:],np.array([[p1[2]],[p1[4]]])) - pm[2]*p1[3]
            
            Gi[1,0] = np.dot(MC[1,:],np.array([[p3[2]],[p3[4]]]))
            
            Fi[0,0] = (Lm[0]*pm[3]-Lm[2]*pm[2])*fm*tm
            
            sol = np.dot(np.linalg.inv(MC),(Gi+Fi))
            
            p4[2] = sol[0]
            
            p4[4] = p3[5]/self.q_ion - 0.5 * (p4[2]**2) / q_m
            
            iter_err = abs(p4[0]-Z)+abs(p4[1]-R)+abs(p4[2]-U)+abs(p4[3]-V)+abs(p4[4]-PHI)
        
            if ( iter_err < corrector_tol ):
            
                break
        
            else:
            
                Z = p4[0] 
                R = p4[1] 
                U = p4[2] 
                V = p4[3] 
                PHI = p4[4]
            
            if iteration == max_iter:
        
                print('Convergence was not achieved')
                
            p4[5] = self.q_ion*p4[4]  + 0.5*self.m_ion*(p4[2]**2+p4[3]**2)
        
            p4[6] = simple_species.n(self,self.n[0],self.T_0,p4[4],self.Gamma,-self.q_ion)
            
            p4[7]= simple_species.temp(self,p4[6],self.n[0],self.T_0,self.Gamma)
            
            p4[8] = np.sqrt(self.Gamma*p4[7]/self.m_ion)
        
            return p4
            
            
"HELPER FUNCTIONS FOR MOC CLASS "
        
def charline(char_type,z,r,u,v,c):
    
    "Returns a struct with the charline, given the data for a point of the charline."
        
    if char_type is 'p':
        
        sqrtp = np.sqrt(u**2+v**2-c**2)
        
        Lp = np.zeros(4)
        
        Lp[0] = u/c-v/sqrtp
        
        Lp[1] = z
        
        Lp[2] = v/c+u/sqrtp
        
        Lp[3] = r
        
        n = np.sqrt(Lp[0]**2 + Lp[2]**2)
        
        Lp[0] = Lp[0]/n
        
        Lp[2] = Lp[2]/n
        
        return Lp
        
    elif char_type is 'm':
        
        sqrtp = np.sqrt(u**2+v**2-c**2)
        
        Lm = np.zeros(4)
        
        Lm[0] = u/c+v/sqrtp
        
        Lm[1] = z
        
        Lm[2] = v/c-u/sqrtp
        
        Lm[3] = r
        
        n = np.sqrt(Lm[0]**2 + Lm[2]**2)
        
        Lm[0] = Lm[0]/n
        
        Lm[2] = Lm[2]/n
        
        return Lm
        
    else:
        
        n = np.sqrt(u**2+v**2)
        
        Lo = np.zeros(4)
        
        Lo[0] = u/n
        
        Lo[1] = z
        
        Lo[2]= v/n
        
        Lo[3] = r
        
        return Lo          
    
                
if __name__ == '__main__':  # When, run for testing only

    P = simple_plasma(1.6e-19,2.1801714e-25,2.1801714e-19,1)
    
    r_0 = np.linspace(0,3,100)
    
    n0 =  np.exp(-6.1505/2*r_0**2)

    Plume = MOC(P,r_0,n0,np.linspace(20000,20000,100),np.linspace(0,12000,100))
    
    Plume.solver()
    
    """z_target = np.array([15,20,25,30])
    
    r_target = np.array([20,25,30,35])
    
    n,u_z,u_r,T,phi = Plume.query(z_target,r_target)
    
    Plume.plot(z_target,r_target,'u_z',[0.8,0.81,0.82,0.83,0.84])"""


          
        