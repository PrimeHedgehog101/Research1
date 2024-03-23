import numpy as np
import csv
from scipy import integrate
from scipy.integrate import solve_bvp

n = 6
s_0 = 0.3 # Sigma_0 
t = np.linspace(0,np.pi/2,400)
n2 = list(np.linspace(1,n,n))
ll2=[]


fl = open('Eigvalues.csv','w',newline='')
writer = csv.writer(fl)

fl2 = open('Eigvalues_p.csv','w',newline='')
writer2 = csv.writer(fl2)


points = 8
sn = list(np.linspace(1,s_0,points)) # setup for param continuation
r = np.zeros(points+1)

for n in n2:
    
    r[0] = (64*n**6 - 32*n**4 + 4*n**2)/(1+4*n**2) # Initial guess for eigenvalue
    y = np.zeros((6, t.size))

    for i in range(0,len(t)):
        y[0,i] = 2/(np.sqrt(np.pi*(1+4*n**2))) * np.sin(2*n*i) # initial guess for eigensolution

    for s_0 in sn :
        
        i = sn.index(s_0)
        
        def E_int(): # Eliptical intergral
            E_int = np.sqrt(1-((np.sin(t))**2/(np.cosh(s_0))**2))
            return integrate.trapezoid(E_int,t)
    
        c = np.pi/2 * (1/np.cosh(s_0)) * (1/E_int()) 
        A = -(c**4*(np.sinh(2*s_0))**2)/4
        
        def Eig_fun(t,y,p):
            k = p[0]
            L_0 = (3*np.sin(2*t)*((np.cosh(2*s_0)**2)-5+4*np.cosh(2*s_0)*np.cos(2*t)))/ (np.cosh(2*s_0)-np.cos(2*t))**3
            
            L_1= -((2*(np.cos(2*t))**2)+8*np.cosh(2*s_0)*np.cos(2*t)-9-
                    (np.cosh(2*s_0))**2)/((np.cosh(2*s_0)-np.cos(2*t))**2)
            
            L_2 = -(3*np.sin(2*t))/(np.cosh(2*s_0)-np.cos(2*t))
    
            return np.vstack((y[1],y[2],y[3],y[4],y[5], k*A*(-(((np.cosh(2*s_0)-np.cos(2*t))**2)/((np.sinh(2*s_0))**2))*y[2]-
                ((3*(np.cosh(2*s_0)-np.cos(2*t))*np.sin(2*t))/(np.sinh(2*s_0)**2))*y[1]+
                ((2*np.sinh(2*s_0)**2 + 3*(np.sin(2*t))**2 - (np.cosh(2*s_0)-np.cos(2*t))**2)/
                ((np.sinh(2*s_0))**2))*y[0])-L_2*y[5]-(L_1+1)*y[4]-(L_0+L_2)*y[3]-
                L_1*y[2]-L_0*y[1]))
        
        def Eig_bc(ya, yb, p):
            return np.array([ya[0],ya[2],ya[4],yb[0],yb[2],yb[4],ya[1]-1])
        
        def eigen_solver():
            return solve_bvp(Eig_fun, Eig_bc, t, y, p=[r[i]])
        
        def y_n():
            return eigen_solver().sol(t)[0]
        y_ntest = y_n()
    
        def Y_n_Evalue():
            return eigen_solver().p[0]  
        
        y_Etest = Y_n_Evalue()
    
        y_plot = y_n()
        
        for j in range(0,len(t)):
            y[0,j] = y_plot[j]
            
        r[i+1] =  Y_n_Evalue()
       
        def y_prime():
            return eigen_solver().sol(t)[1]    
    
        def y_primeprime():
            return eigen_solver().sol(t)[2] 
        
        def y_primeprime2():
            return eigen_solver().sol(t)[3] 
    
        y_n = y_n()
        y_p = y_prime()
        y_pp = y_primeprime()
    
        h = c*(0.5*np.cosh(2*s_0)-0.5*np.cos(2*t))**0.5 
    
        def Norm(): # Normalistion const
            
            F_y = (np.tanh(2*s_0))**2 * (((np.sinh(2*s_0)**2 + 2*np.sin(2*t)**2 - 2*np.cos(2*t)**2 + 2*np.cosh(2*s_0)*np.cos(2*t))*y_plot)/np.sinh(2*s_0)**2 - (-np.cosh(2*s_0) + np.cos(2*t))**2*y_pp/np.sinh(2*s_0)**2 + (3*np.cos(2*t)*np.sin(2*t) - 3*np.sin(2*t)*np.cosh(2*s_0))*y_p/np.sinh(2*s_0)**2)
            trap = 1/h * y_plot * F_y
            gamma = integrate.trapezoid(trap,t)
            Norm_Const = (1/np.sqrt(gamma))
            return Norm_Const
    
        Norm_Const = Norm()
        Y_n = y_n * Norm_Const 
        Y_np = y_p * Norm_Const 
        #writer3.writerow(Y_n)
        

    writer.writerow(y_ntest*Norm_Const)
    writer2.writerow(Y_np)
    #writer.writerow(y_Etest)
    
    ll2.append(y_Etest)
    # ll4.append(y_ntest*Norm_Const)
    
writer.writerow(ll2)    
fl.close()
fl2.close()
