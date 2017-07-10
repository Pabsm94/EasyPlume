# -*- coding: utf-8 -*-
"""
Created on Mon May  8 18:16:01 2017

@author: pablo
"""

import os 

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

import sys
 
sys.path.append(dir_path) #change to src

from src import np,math,interp1d,Hyperplume,SSM,AEM,griddata,interp2d,plt

import time


"""
========================= SSM ELECTRIC POTENTIAL ====================================================0
"""

"""Z_span = np.linspace(0,110,5000)
    
eta_0 = np.linspace(0,10,5000)

n0 =  np.exp(-6.15/2 * eta_0**2)

d0=0.2
    
M0 =20

z = np.linspace(0,20,100)

gamma = [1,1.2]

color = ['red','black']

style= ['-','--']

for (i,gamma_value) in enumerate(gamma):
    
    Plasma = Hyperplume().simple_plasma(1.6e-19,2.1801714e-25,2.1801714e-19,gamma_value)

    eta = [0,1]
    
    Plume = SSM(Plasma,M0,d0,Z_span,eta_0,n0)

    Plume.solver()
    
    for (j,eta_value) in enumerate(eta):
        
        n = Plume.n0[0] * Plume.nu_interp(eta_value) * 1/Plume.h_interp(z)**2
        
        phi = Plume.phi(n,Plume.n0[0],Plume.T_0,Plume.Gamma,Plume.q_ion)*1.6e-19/2.1801714e-19
    
        plt.plot(z,phi,label=r'$\gamma =$' + str(gamma_value) + r'$,\eta =$' + str(eta_value),linestyle=style[j],color=color[i])

plt.plot(z,-gamma[1]/(gamma[1]-1)*np.ones(len(z)),color='blue',linestyle='dotted',label = r'$\tilde{\phi}_{\infty}(\eta=1.2)$') 
   
plt.ylabel(r'$\tilde{\phi}$')

plt.ylim([-10,0])

plt.xlabel(r'$\tilde{z}$')

plt.legend(loc='upper right',prop={'size':10})

plt.title( r'$\tilde{\phi}=e\phi/T_{e0}$' + ' ')

plt.savefig('SSM_ambipolar_potential_field.png')

plt.show()"""

"""
=======================PLUME AEM ELECTRIC POTENTIAL AEM AND ORBIT2D TO FIND ETA LINES==============================================
"""


def orbit2D(XX,YY,UU,VV,x0,y0,dt,nmax):
    
    x,y = np.zeros(nmax),np.zeros(nmax)
    
    x[0],y[0] = x0,y0
    
    #Iu = interp2d(XX.flatten(),YY.flatten(),UU)
    #Iv = interp2d(XX.flatten(),YY.flatten(),VV)
    for i in range(nmax-1):
        
        #u = Iu(x[i],y[i])
        
        u = griddata(np.array((XX.flatten(),YY.flatten())).T,UU.flatten(),(x[i],y[i]))
        
        #v = Iv(x[i],y[i])
        
        v = griddata(np.array((XX.flatten(),YY.flatten())).T,VV.flatten(),(x[i],y[i]))
    
        x[i+1],y[i+1] = x[i]+u*dt,y[i]+v*dt
        
        if np.isnan(u) or np.isnan(v):
            x = x[0:i+1]
            y = y[0:i+1]
            return x,y
            #break
        
    #return x,y

"""gamma_value = [1,1.2]

Z_span = np.linspace(0,110,100)
    
eta_0 = np.linspace(0,10,100)

n0 =  np.exp(-6.15/2*eta_0**2)
uz1 = np.linspace(20000,20000,100)
ur1 = np.linspace(0,40000,100)

phi = []

for gamma in gamma_value:
    
    Plasma = Hyperplume().simple_plasma(1.6e-19,2.1801714e-25,2.1801714e-19,gamma)
    
    Plume1 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
        
    Plume1.solver() 
    
    eta_lines = [0,1]
    
    fact  = 2
    dt    = 1/fact*(np.max(Plume1.z_span)/len(Plume1.z_span))
    zeds  = np.zeros((3,10*fact*len(Plume1.z_span)+1,len(eta_lines)))
    radii = np.zeros((3,10*fact*len(Plume1.z_span)+1,len(eta_lines)))
    nmax = 5*fact*len(Plume1.z_span)
    
    for order in range(Plume1.order+1):
        for i in range(len(eta_lines)):
            z0,r0 = 0,eta_lines[i]
            zeds2,radii2 = orbit2D(Plume1.z_grid,Plume1.r_grid,Plume1.uz[order,:,:]/Plume1.uz0[0],Plume1.ur[order,:,:]/Plume1.uz0[0],z0,r0,dt,nmax)
            num = zeds2.shape[0]
            zeds[order,0,i]=num
            zeds[order, 1:num+1,i]=zeds2
            radii[order,0,i]=num
            radii[order, 1:num+1,i]=radii2
            
    for i in range(len(eta_lines)):
        
        temp = griddata(np.array((Plume1.z_grid.flatten(),Plume1.r_grid.flatten())).T,Plume1.phi[2,:,:].flatten(),(zeds[2,1:,i],radii[2,1:,i]))
    
        phi.append(temp)
    
        
z=np.linspace(0,100,100)
fig,ax = plt.subplots()
ax.plot(zeds[2,1:199,0],phi[0][0:198],color='red',linestyle='solid',label=r'$\gamma=1,\eta=0$') 
ax.plot(zeds[2,1:197,1],phi[1][0:196],color='red',linestyle='dashed',label=r'$\gamma=1,\eta=1$')
ax.plot(zeds[2,1:199,0],phi[2][0:198],color='black',linestyle='solid',label=r'$\gamma=1.2,\eta=0$') 
ax.plot(zeds[2,1:197,1],phi[3][0:196],color='black',linestyle='dashed',label=r'$\gamma=1.2,\eta=1$') 
ax.plot(z,-gamma_value[1]/(gamma_value[1]-1)*np.ones(len(z)),color='blue',linestyle='dotted',label = r'$\tilde{\phi}_{\infty}(\eta=1.2)$')
ax.set_ylim([-10,0])
ax.set_xlim([0,100])
ax.set_ylabel(r'$\tilde{\phi}$')
ax.set_xlabel(r'$\tilde{z}$')
plt.legend(loc='upper right',prop={'size':10})
ax.set_title( r'$\tilde{\phi}=e\phi/T_{e0}$' + ' ')
#ZZ,RR =np.linspace(0,10,100),np.linspace(0,5,100)
#lnn2,u_z,u_r,T,phi,etaAEM = Plume1.query(ZZ,RR)"""
        
"""
=========================== SSM AND AEM PLUME DIVERGENCE ANGLE==========================================================================
"""
"""
Z_span = np.linspace(0,110,100)   
eta_0 = np.linspace(0,10,100)
n0 =  np.exp(-6.15/2 * eta_0**2)

M0_list = np.linspace(5,50,10)
alpha0_list = np.linspace(5,30,10)
d0_list = np.tan(np.radians(alpha0_list))
gamma_list = [1,1.2]

z_f=70
alpha_f_SSM=np.zeros((len(gamma_list),len(d0_list),len(M0_list)))
alpha_f_AEM=np.zeros((len(gamma_list),len(d0_list),len(M0_list)))

for i,gamma in enumerate(gamma_list):
    Plasma = Hyperplume().simple_plasma(1.6e-19,2.1801714e-25,2.1801714e-19,gamma)
    for j,M0 in enumerate(M0_list):
        for k,d0 in enumerate(d0_list):
            
            Plume = SSM(Plasma,M0,d0,Z_span,eta_0,n0)
            Plume.solver()
            
            r_f = Plume.h_interp(z_f)            
            alpha_f_SSM[i,k,j] = math.degrees(math.atan((r_f-1)/z_f))
            

for i,gamma in enumerate(gamma_list):
    Plasma = Hyperplume().simple_plasma(1.6e-19,2.1801714e-25,2.1801714e-19,gamma)
    for j,M0 in enumerate(M0_list):
        UZ0 =  np.sqrt(2.1801714e-19*1*M0**2/(2.1801714e-25))
        uz0 = np.linspace(UZ0,UZ0,len(eta_0))
        for k,d0 in enumerate(d0_list):
            ur0 = np.linspace(0,UZ0*eta_0[-1]*d0,len(eta_0))
            Plume1 = AEM(Plasma,Z_span,eta_0,n0,uz0,ur0,2)
            Plume1.solver()
            fact  = 2
            dt    = 1/fact*(np.max(Plume1.z_span)/len(Plume1.z_span))
            zeds  = np.zeros((3,10*fact*len(Plume1.z_span)+1,1))
            radii = np.zeros((3,10*fact*len(Plume1.z_span)+1,1))
            nmax = 5*fact*len(Plume1.z_span)
            z0,r0 = 0,1
            zeds2,radii2 = orbit2D(Plume1.z_grid,Plume1.r_grid,Plume1.uz[2,:,:]/Plume1.uz0[0],Plume1.ur[2,:,:]/Plume1.uz0[0],z0,r0,dt,nmax)
            try:
                r_f = interp1d(zeds2,radii2)(z_f)
                alpha_f_AEM[i,k,j] = math.degrees(math.atan((r_f-1)/z_f))
            except ValueError:
                alpha_f_AEM[i,j,k] = -1
            
Mach,Alpha = np.meshgrid(M0_list,alpha0_list)

fig,ax = plt.subplots()
cS = plt.contour(Mach,Alpha,alpha_f_SSM[0,:,:],levels=[10,15,20,25,30,35,40],linestyles='solid',colors='red',label='SSM')
cA = plt.contour(Mach,Alpha,alpha_f_AEM[0,:,:],levels=[10,15,20,25,30,35,40],linestyles='solid',colors='blue',label='AEMM')
plt.clabel(cS,cS.levels,fontsize=10,colors='black')
ax.set_title(r'$\alpha_{FR}(deg), \gamma = 1$')
ax.set_xlabel(r'$ M_{0} $')
ax.set_ylabel(r'$\alpha_{0} (deg) $')
plt.legend(loc='best')
plt.savefig('far_plume_divergence_comp_gamma_1.png')

fig,ax = plt.subplots()
cS = plt.contour(Mach,Alpha,alpha_f_SSM[1,:,:],levels=[10,15,20,25,30,35,40],linestyles='solid',colors='red',label='SSM')
cA = plt.contour(Mach,Alpha,alpha_f_AEM[1,:,:],levels=[10,15,20,25,30,35,40],linestyles='solid',colors='blue',label='AEMM')
plt.clabel(cS,cS.levels,fontsize=10,colors='black')
ax.set_title(r'$\alpha_{FR}(deg), \gamma = 1.2$')
ax.set_xlabel(r'$ M_{0} $')
ax.set_ylabel(r'$\alpha_{0} (deg) $')
plt.legend(loc='best')
plt.savefig('far_plume_divergence_comp_gamma_1.png')"""

"""
========== COMPARISON AEM SOLVER AND MARCHING SOLVER================ 
"""
"""Plasma = Hyperplume().simple_plasma(1.6e-19,2.1801714e-25,2.1801714e-19,1)

Z_span = np.linspace(0,100,2001)

eta_0 = np.linspace(0,3,101)
    
n0 =  np.exp(-6.15/2*eta_0**2)

uz1 = np.linspace(20000,20000,101)
ur1 = np.linspace(0,2.183821405597220e+04,101)

Plume = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)

Plume.marching_solver(100) 

Plume1 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
    
Plume1.solver() 

log10_n1=np.log10(np.exp(Plume1.lnn[2,:,:]))

log10_n=np.log10(np.exp(Plume.lnn))

eta_lines = [0.5263157894736842,1]

fact  = 2
dt    = 1/fact*(np.max(Plume1.z_span)/len(Plume1.z_span))
zeds  = np.zeros((10*fact*len(Plume1.z_span)+1,len(eta_lines)))
radii = np.zeros((10*fact*len(Plume1.z_span)+1,len(eta_lines)))
nmax = 5*fact*len(Plume1.z_span)

for i in range(len(eta_lines)):
    z0,r0 = 0,eta_lines[i]
    zeds2,radii2 = orbit2D(Plume1.z_grid,Plume1.r_grid,Plume1.uz[2,:,:]/Plume1.uz0[0],Plume1.ur[2,:,:]/Plume1.uz0[0],z0,r0,dt,nmax)
    num = zeds2.shape[0]
    zeds[0,i]=num
    zeds[1:num+1,i]=zeds2
    radii[0,i]=num
    radii[1:num+1,i]=radii2
        
fig,ax = plt.subplots()

c1 = plt.contour(Plume.z_grid,Plume.r_grid,log10_n,levels=[-6,-5,-4,-3,-2,-1],colors='r',linestyles='dashed')
c2 = plt.contour(Plume1.z_grid,Plume1.r_grid,log10_n1,levels=[-6,-5,-4,-3,-2,-1],colors='r',linestyles='solid')

fmt = {}
strs = ['-6', '-5', '-4', '-3', '-2', '-1']
for l, s in zip(c1.levels, strs):
    fmt[l] = s

label=plt.clabel(c1,c1.levels,colors='black',fontsize=11,fmt=fmt,manual=[(60,50),(50,30),(70,25),(50,10),(20,5),(5,0)])

for l in label:
        l.set_rotation(0)  

#ax.plot(zeds[1:,1],radii[1:,1],linestyle='solid',color='black',label=r'$95 % ion stream line,\eta=1$')
#ax.plot(zeds[1:,0],radii[1:,0],linestyle='solid',color='black',label=r'$50 % ion stream line,\eta\simeq 0.526$')
ax.set_title(r'$log_{10}(n),M_{0}=20,\alpha_{0}=20,\gamma=1$')
ax.set_ylabel(r'$\tilde{r}$')
ax.set_xlabel(r'$\tilde{z}$')
ax.set_ylim([0,50])
ax.set_xlim([0,100])

plt.savefig('AEM_Marching_Solver_comparison.png',bbox_inches='tight')

Plasma = Hyperplume().simple_plasma(1.6e-19,2.1801714e-25,2.1801714e-19,1.2)

Z_span = np.linspace(0,100,2001)

eta_0 = np.linspace(0,3,101)
    
n0 =  np.exp(-6.15/2*eta_0**2)

uz1 = np.linspace(20000,20000,101)
ur1 = np.linspace(0,1.607695154586736e+04,101)

Plume = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)

Plume.marching_solver(100) 

log10_n=np.log10(np.exp(Plume.lnn))
        
fig,ax1 = plt.subplots()

c1 = plt.contourf(Plume.z_grid,Plume.r_grid,log10_n,levels=np.linspace(-6,0,100),linestyles='dashed')
ax1.set_title(r'$log_{10}(n),M_{0}=20,\alpha_{0}=15,\gamma=1.2$')
ax1.set_ylabel(r'$\tilde{r}$')
ax1.set_xlabel(r'$\tilde{z}$')
ax1.set_ylim([0,50])
ax1.set_xlim([0,100])
cbar = fig.colorbar(c1, ticks=[-6,-4,-2,0])
plt.savefig('AEM_Marching_Solver_Results.png',bbox_inches='tight')

gamma_list = [1,1.2]
M0_list = [20,30,40]
alpha_0=[15,20]
d0_list = np.tan(np.radians(alpha_0))
Z_span = np.linspace(0,100,2001)
eta_0 = np.linspace(0,3,101)
n0 =  np.exp(-6.15/2*eta_0**2)
error1 = {}
for i,gamma in enumerate(gamma_list):
    
    Plasma = Hyperplume().simple_plasma(1.6e-19,2.1801714e-25,2.1801714e-19,gamma)
    
    for j,M0 in enumerate(M0_list):
        
        ui0 = np.sqrt(Plasma['Electrons']['Gamma']*Plasma['Electrons']['T_0_electron']*M0**2/Plasma['Ions']['mass_ion'])
    
        uz1 = np.linspace(ui0,ui0,101)
        
        for k,d0 in enumerate(d0_list):
        
            ur1 = np.linspace(0,ui0*eta_0[-1]*d0,101)

            Plume = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
            
            Plume.marching_solver(100)
            
            z,r = np.linspace(0,100,50),np.zeros(50)
            
            grid_points = np.array((Plume.z_grid.flatten(),Plume.r_grid.flatten())).T 
            lnn_marching1 = griddata(grid_points,Plume.lnn[:,:].flatten(),(z,r),method='linear')
            
            Plume1 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
            
            Plume1.solver() 
            
            grid_points = np.array((Plume1.z_grid.flatten(),Plume1.r_grid.flatten())).T 
            lnn_solver1 = griddata(grid_points,Plume1.lnn[2,:,:].flatten(),(z,r),method='linear')
            
            error1['Gamma='+str(gamma)+';M0='+str(M0)+';a0='+str(alpha_0[k])] = np.abs((lnn_solver1-lnn_marching1)/lnn_marching1)*100

z,r = np.linspace(0,100,50),np.zeros(50)         
fig,ax = plt.subplots()
ax.plot(z,error1['Gamma=1'+';M0=20'+';a0='+str(alpha_0[0])],color='red',label=r'$M_{0}=20,\alpha_{0}=15$')
ax.plot(z,error1['Gamma=1'+';M0=20'+';a0='+str(alpha_0[1])],color='black',label=r'$,M_{0}=20,\alpha_{0}=20$')            
ax.plot(z,error1['Gamma=1'+';M0=30'+';a0='+str(alpha_0[0])],color='blue',label=r'$,M_{0}=30,\alpha_{0}=15$')
ax.plot(z,error1['Gamma=1'+';M0=30'+';a0='+str(alpha_0[1])],color='green',label=r'$M_{0}=30,\alpha_{0}=20$')
ax.plot(z,error1['Gamma=1'+';M0=40'+';a0='+str(alpha_0[0])],color='yellow',label=r'$M_{0}=40,\alpha_{0}=15$')           
ax.plot(z,error1['Gamma=1'+';M0=40'+';a0='+str(alpha_0[1])],color='magenta',label=r'$M_{0}=40,\alpha_{0}=20$')
ax.set_ylabel('Relative density error [%]')
ax.set_xlabel('z')
ax.set_title('Plume centerline'+' '+r'$r=0,\gamma=1$')
plt.legend(loc='best')
plt.savefig('AEM_marching_solver_error_1.png')

fig,ax = plt.subplots()
ax.plot(z,error1['Gamma=1.2'+';M0=20'+';a0='+str(alpha_0[0])],color='red',label=r'$M_{0}=20,\alpha_{0}=15$')
ax.plot(z,error1['Gamma=1.2'+';M0=20'+';a0='+str(alpha_0[1])],color='black',label=r'$M_{0}=20,\alpha_{0}=20$')            
ax.plot(z,error1['Gamma=1.2'+';M0=30'+';a0='+str(alpha_0[0])],color='blue',label=r'$M_{0}=30,\alpha_{0}=15$')
ax.plot(z,error1['Gamma=1.2'+';M0=30'+';a0='+str(alpha_0[1])],color='green',label=r'$M_{0}=30,\alpha_{0}=20$')
ax.plot(z,error1['Gamma=1.2'+';M0=40'+';a0='+str(alpha_0[0])],color='yellow',label=r'$M_{0}=40,\alpha_{0}=15$')           
ax.plot(z,error1['Gamma=1.2'+';M0=40'+';a0='+str(alpha_0[1])],color='magenta',label=r'$M_{0}=40,\alpha_{0}=20$')
ax.set_ylabel('Relative density error [%]')
ax.set_xlabel('z')
ax.set_title('Plume centerline'+' '+r'$r=0,\gamma=1.2$')
plt.legend(loc='best')
plt.savefig('AEM_marching_solver_error_12.png')



=================MORE===================
============================================
==================COMPARISONS================
==============================================


gamma_list = [1,1.2]
M0_list = [20,30]
alpha_0=[15]
d0_list = np.tan(np.radians(alpha_0))
Z_span = np.linspace(0,100,2001)
eta_0 = np.linspace(0,3,101)
n0 =  np.exp(-6.15/2*eta_0**2)
error0 = {}
error2 = {}
errorSSM={}
for i,gamma in enumerate(gamma_list):
    
    Plasma = Hyperplume().simple_plasma(1.6e-19,2.1801714e-25,2.1801714e-19,gamma)
    
    for j,M0 in enumerate(M0_list):
        
        ui0 = np.sqrt(Plasma['Electrons']['Gamma']*Plasma['Electrons']['T_0_electron']*M0**2/Plasma['Ions']['mass_ion'])
    
        uz1 = np.linspace(ui0,ui0,101)
        
        for k,d0 in enumerate(d0_list):
        
            ur1 = np.linspace(0,ui0*eta_0[-1]*d0,101)

            Plume = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
            
            Plume.marching_solver(100)
            
            z,r = 60*np.ones(50),np.linspace(0,50,50)
            
            grid_points = np.array((Plume.z_grid.flatten(),Plume.r_grid.flatten())).T 
            lnn_marching1 = griddata(grid_points,Plume.lnn[:,:].flatten(),(z,r),method='linear')
            
            Plume1 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
            
            Plume1.solver() 
            
            grid_points = np.array((Plume1.z_grid.flatten(),Plume1.r_grid.flatten())).T 
            lnn_solver0 = griddata(grid_points,Plume1.lnn[0,:,:].flatten(),(z,r),method='linear')
            lnn_solver2 = griddata(grid_points,Plume1.lnn[2,:,:].flatten(),(z,r),method='linear')
            
            error0['Gamma='+str(gamma)+';M0='+str(M0)+';a0='+str(alpha_0[k])] = np.abs((lnn_solver2-lnn_marching1)/lnn_marching1)
            error2['Gamma='+str(gamma)+';M0='+str(M0)+';a0='+str(alpha_0[k])] = np.abs((lnn_solver0-lnn_marching1)/lnn_marching1)
            
            Plume = SSM(Plasma,M0,d0,Z_span,eta_0,n0)

            Plume.solver()

            n,u_z,u_r,T,phi,error,etaSSM = Plume.query(z,r)
            
            errorSSM['Gamma='+str(gamma)+';M0='+str(M0)+';a0='+str(alpha_0[k])] = np.abs((np.log(n)-lnn_marching1)/lnn_marching1)
            
r=np.linspace(0,50,50)   
fig,ax = plt.subplots()
ax.plot(r,error0['Gamma=1'+';M0=20'+';a0='+str(alpha_0[0])],color='blck',label=r'$M_{0}=20,\alpha_{0}=15$')
ax.plot(r,error2['Gamma=1'+';M0=20'+';a0='+str(alpha_0[0])],color='black',label=r'$,M_{0}=20,\alpha_{0}=15$')
ax.plot(r,errorSSM['Gamma=1'+';M0=20'+';a0='+str(alpha_0[0])],color='black',label=r'$,M_{0}=20,\alpha_{0}=15$')             
ax.set_ylabel('Relative density error [%]')
ax.set_xlabel('\tilde{r}')
ax.set_title('Relative errors'+' '+r'$M_{0}=20,\gamma=1$')
plt.legend(loc='best')
plt.savefig('full_relative_error_comp_M0_20_gamma_1.png')

fig,ax = plt.subplots()
ax.plot(r,error0['Gamma=1.2'+';M0=20'+';a0='+str(alpha_0[0])],color='blck',label=r'$M_{0}=20,\alpha_{0}=15$')
ax.plot(r,error2['Gamma=1.2'+';M0=20'+';a0='+str(alpha_0[0])],color='black',label=r'$,M_{0}=20,\alpha_{0}=15$')
ax.plot(r,errorSSM['Gamma=1.2'+';M0=20'+';a0='+str(alpha_0[0])],color='black',label=r'$,M_{0}=20,\alpha_{0}=15$')             
ax.set_ylabel('Relative density error')
ax.set_xlabel('\tilde{r}')
ax.set_title('Relative errors'+' '+r'$M_{0}=20,\gamma=1.2$')
plt.legend(loc='best')
plt.savefig('full_relative_error_comp_M0_20_gamma_1.2.png')

fig,ax = plt.subplots()
ax.plot(r,error0['Gamma=1'+';M0=30'+';a0='+str(alpha_0[0])],color='blck',label=r'$M_{0}=30,\alpha_{0}=15$')
ax.plot(r,error2['Gamma=1'+';M0=30'+';a0='+str(alpha_0[0])],color='black',label=r'$,M_{0}=30,\alpha_{0}=15$')
ax.plot(r,errorSSM['Gamma=1'+';M0=30'+';a0='+str(alpha_0[0])],color='black',label=r'$,M_{0}=30,\alpha_{0}=15$')             
ax.set_ylabel('Relative density error')
ax.set_xlabel('\tilde{r}')
ax.set_title('Relative errors'+' '+r'$M_{0}=30,\gamma=1$')
plt.legend(loc='best')
plt.savefig('full_relative_error_comp_M0_30_gamma_1.png')

fig,ax = plt.subplots()
ax.plot(r,error0['Gamma=1.2'+';M0=30'+';a0='+str(alpha_0[0])],color='blck',label=r'$M_{0}=30,\alpha_{0}=15$')
ax.plot(r,error2['Gamma=1'+';M0=30'+';a0='+str(alpha_0[0])],color='black',label=r'$,M_{0}=30,\alpha_{0}=15$')
ax.plot(r,errorSSM['Gamma=1'+';M0=30'+';a0='+str(alpha_0[0])],color='black',label=r'$,M_{0}=30,\alpha_{0}=15$')             
ax.set_ylabel('Relative density error [%]')
ax.set_xlabel('\tilde{r}')
ax.set_title('Relative errors'+' '+r'$M_{0}=30,\gamma=1.2$')
plt.legend(loc='best')
plt.savefig('full_relative_error_comp_M0_30_gamma_1.2.png')"""

"======================== COMPARISON AEM SSM==========================================================="

"""Plasma = Hyperplume().simple_plasma(1.6e-19,2.1801714e-25,2.1801714e-19,1)
    
Z_span = np.linspace(0,100,5000)

eta_0 = np.linspace(0,100,5000)

n0 =  np.exp(-6.15/2 * eta_0**2)

d0=0.2679491924311227

M0 =20

Plume = SSM(Plasma,M0,d0,Z_span,eta_0,n0)

ZZ,RR = np.meshgrid(np.linspace(0,100,2001),np.linspace(0,100,101))

Plume.solver()

n1,u_z,u_r,T,phi,error,etaSSM = Plume.query(ZZ,RR)

z=np.linspace(0,100,2001)

r95_SSM=Plume.h_interp(z)

r50_SSM=Plume.h_interp(z)*0.5263157894736842

lnn_SSM = np.log10(n1)
    
Z_span = np.linspace(0,100,2001)
    
eta_0 = np.linspace(0,3,101)

n0 =  np.exp(-6.15/2*eta_0**2)

uz1 = np.linspace(20000,20000,101)
ur1 = np.linspace(0,1.607695154586736e+04,101)

Plume1 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)

Plume1.solver() 

lnn0_AEM = np.log10(np.exp(Plume1.lnn[0,:,:]))
lnn1_AEM = np.log10(np.exp(Plume1.lnn[1,:,:]))
lnn2_AEM = np.log10(np.exp(Plume1.lnn[2,:,:]))


Z_span = np.linspace(0,100,100)
    
eta_0 = np.linspace(0,3,100)

n0 =  np.exp(-6.15/2*eta_0**2)

uz1 = np.linspace(20000,20000,100)
ur1 = np.linspace(0,1.607695154586736e+04,100)
Plume3 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
Plume3.solver()

fact  = 2
eta_lines = [0.5263157894736842,1]
dt    = 1/fact*(np.max(Plume3.z_span)/len(Plume3.z_span))
nmax = 5*fact*len(Plume3.z_span)
zeds=np.zeros((3,nmax,2))
radii=np.zeros((3,nmax,2))
start_time = time.time()

for order in range(Plume3.order+1):
    for i in range(len(eta_lines)):
        z0,r0 = 0,eta_lines[i]
        zeds2,radii2 = orbit2D(Plume3.z_grid,Plume3.r_grid,Plume3.uz[order,:,:]/Plume3.uz0[0],Plume3.ur[order,:,:]/Plume3.uz0[0],z0,r0,dt,nmax)
        zeds[order,0:zeds2.shape[0],i]=zeds2
        radii[order,0:radii2.shape[0],i]=radii2
print("--- %s seconds ---" % (time.time() - start_time))


Z_span = np.linspace(0,100,2001)
    
eta_0 = np.linspace(0,3,101)

n0 =  np.exp(-6.15/2*eta_0**2)

uz1 = np.linspace(20000,20000,101)
ur1 = np.linspace(0,1.607695154586736e+04,101)
Plume2 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
    
Plume2.marching_solver(100)

lnn_AEM_march = np.log10(np.exp(Plume2.lnn[:,:]))

#zeds20,radii20 = orbit2D(Plume2.z_grid,Plume2.r_grid,Plume2.uz[:,:]/Plume2.uz0[0],Plume2.ur[:,:]/Plume2.uz0[0],0,0.5263157894736842,dt,nmax)
#zeds21,radii21 = orbit2D(Plume2.z_grid,Plume2.r_grid,Plume2.uz[:,:]/Plume2.uz0[0],Plume2.ur[:,:]/Plume2.uz0[0],0,1,dt,nmax)
    

fig,ax = plt.subplots()

cSSM = plt.contour(ZZ,RR,lnn_SSM,levels=[-6,-5,-4,-3,-2,-1],colors='red',linestyles='dotted')

ax.plot(z,r95_SSM,color='black',linestyle='dotted')
ax.plot(z,r50_SSM,color='black',linestyle='dotted')

cAEM2 = plt.contour(Plume1.z_grid,Plume1.r_grid,lnn2_AEM,levels=[-6,-5,-4,-3,-2,-1],colors='red',linestyles='dashed')

ax.plot(zeds[2,0:200,0],radii[2,0:200,0],color='black',linestyle='dashed')
ax.plot(zeds[2,0:202,1],radii[2,0:202,1],color='black',linestyle='dashed')

cAEMM = plt.contour(Plume2.z_grid,Plume2.r_grid,lnn_AEM_march,levels=[-6,-5,-4,-3,-2,-1],colors='red',linestyles='solid')

#ax.plot(zeds20,radii20,colors='black',linestyle='solid',label='eta50%')
#ax.plot(zeds21,radii21,colors='black',linestyle='solid',label='eta95%')

fmt = {}
strs = ['-6', '-5', '-4', '-3', '-2', '-1']
for l, s in zip(cSSM.levels, strs):
    fmt[l] = s

label=plt.clabel(cSSM,cSSM.levels,colors='black',fontsize=11,fmt=fmt,manual=[(60,50),(50,25),(40,18),(40,10),(25,5),(5,0)])

for l in label:
        l.set_rotation(0)  
        

ax.annotate('50%', xy=(80,15),xytext=(80, 17))
ax.annotate('95%', xy=(90,30),xytext=(90, 35))
        
ax.set_ylabel(r'$\tilde{r}$')
ax.set_xlabel(r'$\tilde{z}$')
ax.set_title(r'$log_{10}(\tilde{n}),M_{0}=20,\gamma = 1$')
ax.set_ylim([0,50])
ax.set_xlim([0,100])

plt.savefig('full_comp_M0_20_alpha0_15_gamma_1.png')"""



"""Plasma = Hyperplume().simple_plasma(1.6e-19,2.1801714e-25,2.1801714e-19,1.2)
    
Z_span = np.linspace(0,100,5000)

eta_0 = np.linspace(0,100,5000)

n0 =  np.exp(-6.15/2 * eta_0**2)

d0=0.2679491924311227

M0 =20

Plume = SSM(Plasma,M0,d0,Z_span,eta_0,n0)

ZZ,RR = np.meshgrid(np.linspace(0,100,2001),np.linspace(0,100,101))

Plume.solver()

n1,u_z,u_r,T,phi,error,etaSSM = Plume.query(ZZ,RR)

z=np.linspace(0,100,2001)

r95_SSM=Plume.h_interp(z)

r50_SSM=Plume.h_interp(z)*0.5263157894736842

lnn_SSM = np.log10(n1)
    
Z_span = np.linspace(0,100,2001)
    
eta_0 = np.linspace(0,3,101)

n0 =  np.exp(-6.15/2*eta_0**2)

uz1 = np.linspace(20000,20000,101)
ur1 = np.linspace(0,1.607695154586736e+04,101)

Plume1 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)

Plume1.solver() 

lnn0_AEM = np.log10(np.exp(Plume1.lnn[0,:,:]))
lnn1_AEM = np.log10(np.exp(Plume1.lnn[1,:,:]))
lnn2_AEM = np.log10(np.exp(Plume1.lnn[2,:,:]))


Z_span = np.linspace(0,100,100)
    
eta_0 = np.linspace(0,3,100)

n0 =  np.exp(-6.15/2*eta_0**2)

uz1 = np.linspace(20000,20000,100)
ur1 = np.linspace(0,1.607695154586736e+04,100)
Plume3 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
Plume3.solver()

fact  = 2
eta_lines = [0.5263157894736842,1]
dt    = 1/fact*(np.max(Plume3.z_span)/len(Plume3.z_span))
nmax = 5*fact*len(Plume3.z_span)
zeds=np.zeros((3,nmax,2))
radii=np.zeros((3,nmax,2))
start_time = time.time()

for order in range(Plume3.order+1):
    for i in range(len(eta_lines)):
        z0,r0 = 0,eta_lines[i]
        zeds2,radii2 = orbit2D(Plume3.z_grid,Plume3.r_grid,Plume3.uz[order,:,:]/Plume3.uz0[0],Plume3.ur[order,:,:]/Plume3.uz0[0],z0,r0,dt,nmax)
        zeds[order,0:zeds2.shape[0],i]=zeds2
        radii[order,0:radii2.shape[0],i]=radii2
print("--- %s seconds ---" % (time.time() - start_time))


Z_span = np.linspace(0,100,2001)
    
eta_0 = np.linspace(0,3,101)

n0 =  np.exp(-6.15/2*eta_0**2)

uz1 = np.linspace(20000,20000,101)
ur1 = np.linspace(0,1.607695154586736e+04,101)
Plume2 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
    
Plume2.marching_solver(100)

lnn_AEM_march = np.log10(np.exp(Plume2.lnn[:,:]))

#zeds20,radii20 = orbit2D(Plume2.z_grid,Plume2.r_grid,Plume2.uz[:,:]/Plume2.uz0[0],Plume2.ur[:,:]/Plume2.uz0[0],0,0.5263157894736842,dt,nmax)
#zeds21,radii21 = orbit2D(Plume2.z_grid,Plume2.r_grid,Plume2.uz[:,:]/Plume2.uz0[0],Plume2.ur[:,:]/Plume2.uz0[0],0,1,dt,nmax)
    

fig,ax = plt.subplots()

cSSM = plt.contour(ZZ,RR,lnn_SSM,levels=[-6,-5,-4,-3,-2,-1],colors='red',linestyles='dotted')

ax.plot(z,r95_SSM,color='black',linestyle='dotted')
ax.plot(z,r50_SSM,color='black',linestyle='dotted')

cAEM2 = plt.contour(Plume1.z_grid,Plume1.r_grid,lnn2_AEM,levels=[-6,-5,-4,-3,-2,-1],colors='red',linestyles='dashed')

ax.plot(zeds[2,0:200,0],radii[2,0:200,0],color='black',linestyle='dashed')
ax.plot(zeds[2,0:202,1],radii[2,0:202,1],color='black',linestyle='dashed')

cAEMM = plt.contour(Plume2.z_grid,Plume2.r_grid,lnn_AEM_march,levels=[-6,-5,-4,-3,-2,-1],colors='red',linestyles='solid')

#ax.plot(zeds20,radii20,colors='black',linestyle='solid',label='eta50%')
#ax.plot(zeds21,radii21,colors='black',linestyle='solid',label='eta95%')

fmt = {}
strs = ['-6', '-5', '-4', '-3', '-2', '-1']
for l, s in zip(cSSM.levels, strs):
    fmt[l] = s

label=plt.clabel(cSSM,cSSM.levels,colors='black',fontsize=11,fmt=fmt,manual=[(60,50),(50,25),(40,18),(40,10),(25,5),(5,0)])

for l in label:
        l.set_rotation(0)  
        

ax.annotate('50%', xy=(80,15),xytext=(80, 17))
ax.annotate('95%', xy=(90,30),xytext=(90, 35))
        
ax.set_ylabel(r'$\tilde{r}$')
ax.set_xlabel(r'$\tilde{z}$')
ax.set_title(r'$log_{10}(\tilde{n}),M_{0}=20,\gamma = 1.2$')
ax.set_ylim([0,50])
ax.set_xlim([0,100])

plt.savefig('full_comp_M0_20_alpha0_15_gamma_1.2.png')



Plasma = Hyperplume().simple_plasma(1.6e-19,2.1801714e-25,2.1801714e-19,1)
    
Z_span = np.linspace(0,100,5000)

eta_0 = np.linspace(0,100,5000)

n0 =  np.exp(-6.15/2 * eta_0**2)

d0=0.2679491924311227

M0 =30

Plume = SSM(Plasma,M0,d0,Z_span,eta_0,n0)

ZZ,RR = np.meshgrid(np.linspace(0,100,2001),np.linspace(0,100,101))

Plume.solver()

n1,u_z,u_r,T,phi,error,etaSSM = Plume.query(ZZ,RR)

z=np.linspace(0,100,2001)

r95_SSM=Plume.h_interp(z)

r50_SSM=Plume.h_interp(z)*0.5263157894736842

lnn_SSM = np.log10(n1)
    
Z_span = np.linspace(0,100,2001)
    
eta_0 = np.linspace(0,3,101)

n0 =  np.exp(-6.15/2*eta_0**2)

uz1 = np.linspace(30000,30000,101)
ur1 = np.linspace(0,2.411542731880104e+04,101)

Plume1 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)

Plume1.solver() 

lnn0_AEM = np.log10(np.exp(Plume1.lnn[0,:,:]))
lnn1_AEM = np.log10(np.exp(Plume1.lnn[1,:,:]))
lnn2_AEM = np.log10(np.exp(Plume1.lnn[2,:,:]))


Z_span = np.linspace(0,100,100)
    
eta_0 = np.linspace(0,3,100)

n0 =  np.exp(-6.15/2*eta_0**2)

uz1 = np.linspace(30000,30000,100)
ur1 = np.linspace(0,2.411542731880104e+04,100)
Plume3 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
Plume3.solver()

fact  = 2
eta_lines = [0.5263157894736842,1]
dt    = 1/fact*(np.max(Plume3.z_span)/len(Plume3.z_span))
nmax = 5*fact*len(Plume3.z_span)
zeds=np.zeros((3,nmax,2))
radii=np.zeros((3,nmax,2))
start_time = time.time()

for order in range(Plume3.order+1):
    for i in range(len(eta_lines)):
        z0,r0 = 0,eta_lines[i]
        zeds2,radii2 = orbit2D(Plume3.z_grid,Plume3.r_grid,Plume3.uz[order,:,:]/Plume3.uz0[0],Plume3.ur[order,:,:]/Plume3.uz0[0],z0,r0,dt,nmax)
        zeds[order,0:zeds2.shape[0],i]=zeds2
        radii[order,0:radii2.shape[0],i]=radii2
print("--- %s seconds ---" % (time.time() - start_time))


Z_span = np.linspace(0,100,2001)
    
eta_0 = np.linspace(0,3,101)

n0 =  np.exp(-6.15/2*eta_0**2)

uz1 = np.linspace(30000,30000,101)
ur1 = np.linspace(0,2.411542731880104e+04,101)
Plume2 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
    
Plume2.marching_solver(100)

lnn_AEM_march = np.log10(np.exp(Plume2.lnn[:,:]))

#zeds20,radii20 = orbit2D(Plume2.z_grid,Plume2.r_grid,Plume2.uz[:,:]/Plume2.uz0[0],Plume2.ur[:,:]/Plume2.uz0[0],0,0.5263157894736842,dt,nmax)
#zeds21,radii21 = orbit2D(Plume2.z_grid,Plume2.r_grid,Plume2.uz[:,:]/Plume2.uz0[0],Plume2.ur[:,:]/Plume2.uz0[0],0,1,dt,nmax)
    

fig,ax = plt.subplots()

cSSM = plt.contour(ZZ,RR,lnn_SSM,levels=[-6,-5,-4,-3,-2,-1],colors='red',linestyles='dotted')

ax.plot(z,r95_SSM,color='black',linestyle='dotted')
ax.plot(z,r50_SSM,color='black',linestyle='dotted')

cAEM2 = plt.contour(Plume1.z_grid,Plume1.r_grid,lnn2_AEM,levels=[-6,-5,-4,-3,-2,-1],colors='red',linestyles='dashed')

ax.plot(zeds[2,0:200,0],radii[2,0:200,0],color='black',linestyle='dashed')
ax.plot(zeds[2,0:202,1],radii[2,0:202,1],color='black',linestyle='dashed')

cAEMM = plt.contour(Plume2.z_grid,Plume2.r_grid,lnn_AEM_march,levels=[-6,-5,-4,-3,-2,-1],colors='red',linestyles='solid')

#ax.plot(zeds20,radii20,colors='black',linestyle='solid',label='eta50%')
#ax.plot(zeds21,radii21,colors='black',linestyle='solid',label='eta95%')

fmt = {}
strs = ['-6', '-5', '-4', '-3', '-2', '-1']
for l, s in zip(cSSM.levels, strs):
    fmt[l] = s

label=plt.clabel(cSSM,cSSM.levels,colors='black',fontsize=11,fmt=fmt,manual=[(60,50),(50,25),(40,18),(40,10),(25,5),(5,0)])

for l in label:
        l.set_rotation(0)  
        

ax.annotate('50%', xy=(80,15),xytext=(80, 17))
ax.annotate('95%', xy=(90,30),xytext=(90, 35))
        
ax.set_ylabel(r'$\tilde{r}$')
ax.set_xlabel(r'$\tilde{z}$')
ax.set_title(r'$log_{10}(\tilde{n}),M_{0}=30,\gamma = 1$')
ax.set_ylim([0,50])
ax.set_xlim([0,100])

plt.savefig('full_comp_M0_30_alpha0_15_gamma_1.png')


Plasma = Hyperplume().simple_plasma(1.6e-19,2.1801714e-25,2.1801714e-19,1.2)
    
Z_span = np.linspace(0,100,5000)

eta_0 = np.linspace(0,100,5000)

n0 =  np.exp(-6.15/2 * eta_0**2)

d0=0.2679491924311227

M0 =30

Plume = SSM(Plasma,M0,d0,Z_span,eta_0,n0)

ZZ,RR = np.meshgrid(np.linspace(0,100,2001),np.linspace(0,100,101))

Plume.solver()

n1,u_z,u_r,T,phi,error,etaSSM = Plume.query(ZZ,RR)

z=np.linspace(0,100,2001)

r95_SSM=Plume.h_interp(z)

r50_SSM=Plume.h_interp(z)*0.5263157894736842

lnn_SSM = np.log10(n1)
    
Z_span = np.linspace(0,100,2001)
    
eta_0 = np.linspace(0,3,101)

n0 =  np.exp(-6.15/2*eta_0**2)

uz1 = np.linspace(30000,30000,101)
ur1 = np.linspace(0,2.411542731880104e+04,101)

Plume1 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)

Plume1.solver() 

lnn0_AEM = np.log10(np.exp(Plume1.lnn[0,:,:]))
lnn1_AEM = np.log10(np.exp(Plume1.lnn[1,:,:]))
lnn2_AEM = np.log10(np.exp(Plume1.lnn[2,:,:]))


Z_span = np.linspace(0,100,100)
    
eta_0 = np.linspace(0,3,100)

n0 =  np.exp(-6.15/2*eta_0**2)

uz1 = np.linspace(30000,30000,100)
ur1 = np.linspace(0,2.411542731880104e+04,100)
Plume3 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
Plume3.solver()

fact  = 2
eta_lines = [0.5263157894736842,1]
dt    = 1/fact*(np.max(Plume3.z_span)/len(Plume3.z_span))
nmax = 5*fact*len(Plume3.z_span)
zeds=np.zeros((3,nmax,2))
radii=np.zeros((3,nmax,2))
start_time = time.time()

for order in range(Plume3.order+1):
    for i in range(len(eta_lines)):
        z0,r0 = 0,eta_lines[i]
        zeds2,radii2 = orbit2D(Plume3.z_grid,Plume3.r_grid,Plume3.uz[order,:,:]/Plume3.uz0[0],Plume3.ur[order,:,:]/Plume3.uz0[0],z0,r0,dt,nmax)
        zeds[order,0:zeds2.shape[0],i]=zeds2
        radii[order,0:radii2.shape[0],i]=radii2
print("--- %s seconds ---" % (time.time() - start_time))


Z_span = np.linspace(0,100,2001)
    
eta_0 = np.linspace(0,3,101)

n0 =  np.exp(-6.15/2*eta_0**2)

uz1 = np.linspace(30000,30000,101)
ur1 = np.linspace(0,2.411542731880104e+04,101)
Plume2 = AEM(Plasma,Z_span,eta_0,n0,uz1,ur1,2)
    
Plume2.marching_solver(100)

lnn_AEM_march = np.log10(np.exp(Plume2.lnn[:,:]))

#zeds20,radii20 = orbit2D(Plume2.z_grid,Plume2.r_grid,Plume2.uz[:,:]/Plume2.uz0[0],Plume2.ur[:,:]/Plume2.uz0[0],0,0.5263157894736842,dt,nmax)
#zeds21,radii21 = orbit2D(Plume2.z_grid,Plume2.r_grid,Plume2.uz[:,:]/Plume2.uz0[0],Plume2.ur[:,:]/Plume2.uz0[0],0,1,dt,nmax)
    

fig,ax = plt.subplots()

cSSM = plt.contour(ZZ,RR,lnn_SSM,levels=[-6,-5,-4,-3,-2,-1],colors='red',linestyles='dotted')

ax.plot(z,r95_SSM,color='black',linestyle='dotted')
ax.plot(z,r50_SSM,color='black',linestyle='dotted')

cAEM2 = plt.contour(Plume1.z_grid,Plume1.r_grid,lnn2_AEM,levels=[-6,-5,-4,-3,-2,-1],colors='red',linestyles='dashed')

ax.plot(zeds[2,0:200,0],radii[2,0:200,0],color='black',linestyle='dashed')
ax.plot(zeds[2,0:202,1],radii[2,0:202,1],color='black',linestyle='dashed')

cAEMM = plt.contour(Plume2.z_grid,Plume2.r_grid,lnn_AEM_march,levels=[-6,-5,-4,-3,-2,-1],colors='red',linestyles='solid')

#ax.plot(zeds20,radii20,colors='black',linestyle='solid',label='eta50%')
#ax.plot(zeds21,radii21,colors='black',linestyle='solid',label='eta95%')

fmt = {}
strs = ['-6', '-5', '-4', '-3', '-2', '-1']
for l, s in zip(cSSM.levels, strs):
    fmt[l] = s

label=plt.clabel(cSSM,cSSM.levels,colors='black',fontsize=11,fmt=fmt,manual=[(60,50),(50,25),(40,18),(40,10),(25,5),(5,0)])

for l in label:
        l.set_rotation(0)  
        

ax.annotate('50%', xy=(80,15),xytext=(80, 17))
ax.annotate('95%', xy=(90,30),xytext=(90, 35))
        
ax.set_ylabel(r'$\tilde{r}$')
ax.set_xlabel(r'$\tilde{z}$')
ax.set_title(r'$log_{10}(\tilde{n}),M_{0}=30,\gamma = 1.2$')
ax.set_ylim([0,50])
ax.set_xlim([0,100])

plt.savefig('full_comp_M0_30_alpha0_15_gamma_1.2.png')
"""







