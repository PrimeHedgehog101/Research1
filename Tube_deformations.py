import numpy as np
from scipy import integrate
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt 
import csv
import matplotlib.colors as mcolors
####################################Values#####################################

n   = 1 # Mode
s_0 = 0.6 # Sigma_0 
F   = 1 # Tension
a = 0.85 # 'a' value 
ep = 0.6
t = np.linspace(0,np.pi/2,400)
z = 0.5



#################################Functions to change###########################

def pressure(t):
    return -2*np.cos(2*t)

# def pressure_dt(t):
#     return 10*np.sin(2*t)
#     # return 0.78 - (0.01*np.log(np.cosh(t/(2*0.01) - np.pi/(4*0.01))))

# def pressure(t):
#     mk = list(-100*t[:200])
#     #mk2 = list(t[650]*10*t[650:850]/t[650:850])
#     mk3 = list(-100*((np.pi/2)-t[200:]))
#     return np.array(mk+mk3)

pp = pressure(t)
def pressure_dt(t):
    p2 = pp
    m = len(t)
    derivative = np.zeros(m)

    for i in range(1, m-1):
        derivative[i] = (p2[i+1] - p2[i-1]) / (t[i+1] - t[i-1])

    derivative[0] = (p2[1] - p2[0]) / (t[1] - t[0])
    derivative[m-1] = (p2[m-1] - p2[m-2]) / (t[m-1] - t[m-2])
    return derivative








# def pressure_dt(t):
#     return 4*np.sin(2*t) 

    
    
###########################Extracting data from Eigen data#####################

Eig_data=[]
Eig_data_dt=[]
Eig_data_test = []
Eig_data_2 = []

with open('Eigvalues.csv') as W:
    reader = csv.reader(W,quoting=csv.QUOTE_NONNUMERIC)
    
    for row in reader:
        Eig_data.append(row)
        
with open('Eigvalues_p.csv') as W:
    reader2 = csv.reader(W,quoting=csv.QUOTE_NONNUMERIC)
    
    for row in reader2:
        Eig_data_dt.append(row)
           
        
        
l_n2 = np.array(Eig_data[-1])
Y_n2 = np.array(Eig_data[0:-1])
Y_np2 = np.array(Eig_data_dt)




##############################################################################

n2 = list(np.linspace(1,6,6)) # setting up modes to work over, goes from 1 to n
ll=[]
ll2=[]

# Initialise sums
Q_n_sum = 0
eta_sum = 0 
eta_sum_dx = 0
area_z_sum = 0

###############################################################################
for n in n2:
    index = n2.index(n) # Using the corresponding eig values for a mode n
    l_n = l_n2[index]
    u_n = np.sqrt(l_n/F)
    Y_n = Y_n2[index]
    Y_np = Y_np2[index]
    
    def E_int(): # Eliptical intergral
        E_int = np.sqrt(1-((np.sin(t))**2/(np.cosh(s_0))**2))
        return integrate.trapezoid(E_int,t)
    
    c = np.pi/2 * (1/np.cosh(s_0)) * (1/E_int())
    h = c*(0.5*np.cosh(2*s_0)-0.5*np.cos(2*t))**0.5 
    A_Bar = np.pi*a**2*c**2*np.sinh(2*s_0)/2
    
    
    def Q_n_solver_my(n): #Q_n value
        
        y = (pressure_dt(t) - (pressure_dt(t)*np.cos(2*t)-3*pressure(t)*np.sin(2*t))/(np.cosh(2*s_0)))*Y_n
        Q_n = integrate.trapezoid(y,t) * np.tanh(2*s_0)
        return Q_n
    
    Q_n = Q_n_solver_my(n)
    ll.append(Q_n) # finds Q_n for each n and addds to list to graph
    
    def t_n(n): # t_n value
        const = 24/(np.pi * c**2 * np.sinh(2*s_0)**2)
        integral2 = np.sin(2*t)*Y_n
        return integrate.trapezoid(integral2,t) * const
    
    t_n = t_n(n)
    
    def fun2(x,y): # Set up for alpha_n area change
        return np.vstack((y[1], (u_n**2*(y[0]))-(Q_n*t_n/F)))
            
    def bc2(ya, yb):
        return np.array([ya[0],yb[0]])
    
    def area_z2(z): # alpha_n
        x = np.linspace(0,1,101)
        y_a = np.zeros((2, x.size))
        y_a[0] = 0.1
        res_a = solve_bvp(fun2, bc2, x, y_a)
        x_plot = np.linspace(0, 1, 101)
        y_plot_a = res_a.sol(x_plot)[0]
        return y_plot_a[int(z*100)]
    
    def area_z3(): # alpha_n
        x = np.linspace(0,1,101)
        y_a = np.zeros((2, x.size))
        y_a[0] = 0.1
        res_a = solve_bvp(fun2, bc2, x, y_a)
        x_plot = np.linspace(0, 1, 101)
        y_plot_a = res_a.sol(x_plot)[0]
        return y_plot_a
    
    def eta(z):
        eta =  1/t_n * Y_n * area_z2(z) 
        return eta

    def eta_dx(z):
        eta_dx = 1/t_n * area_z2(z) * Y_np
        return eta_dx
    
    area_test = area_z2(0.5)
    ll2.append(area_test)
    
    eta_n = eta(z) 
    eta_n_dx = eta_dx(z)
    area_z_n = area_z2(z)
    Q_n_sum += Q_n
    eta_sum += eta(z)
    eta_sum_dx += eta_dx(z)
    area_z_sum += area_z2(z)
    area_n_z3 = area_z3()

def xi(z):
    xi = (eta_n*np.sin(2*t) - (2*h**2/c**2*eta_n_dx))/np.sinh(2*s_0)
    return xi

def position_x(z):
    position_x = (a*c*np.cosh(s_0)*np.cos(t)) + ep*a*c/(h**2) * (xi(z) * np.sinh(s_0)*np.cos(t) - eta_n * np.cosh(s_0)*np.sin(t))
    return position_x

def position_y(z):
    position_y = (a*c*np.sinh(s_0)*np.sin(t)) + ep*a*c/(h**2) * (xi(z) * np.cosh(s_0)*np.sin(t) +  eta_n * np.sinh(s_0)*np.cos(t))
    return position_y  



def xi_sum(z):
    xi = (eta_sum*np.sin(2*t) - (2*h**2/c**2*eta_sum_dx))/np.sinh(2*s_0)
    return xi

def position_x_sum(z):
    position_x = (a*c*np.cosh(s_0)*np.cos(t)) + ep*a*c/(h**2) * (xi_sum(z) * np.sinh(s_0)*np.cos(t) - eta_sum * np.cosh(s_0)*np.sin(t))
    return position_x

def position_y_sum(z):
    position_y = (a*c*np.sinh(s_0)*np.sin(t)) + ep*a*c/(h**2) * (xi_sum(z) * np.cosh(s_0)*np.sin(t) +  eta_sum * np.sinh(s_0)*np.cos(t))
    return position_y  



def undeformed_x():
    undeformed_x = a*c*np.cosh(s_0)*np.cos(t)
    return undeformed_x

def undeformed_y():
    undeformed_y = a*c*np.sinh(s_0)*np.sin(t)
    return undeformed_y

###################################Graphing####################################

def p_plot():
    fig=plt.figure()
    ax = plt.gca()
    ax.set_xticks(np.arange(0, np.pi/2 +0.01, np.pi/8))
    labels = ['$0$',r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$',r'$\frac{3\pi}{8}$', r'$\frac{\pi}{2}$']
    ax.set_xticklabels(labels)
    plt.plot(t,pp,color='black')
    #fig.savefig("X1.svg", bbox_inches='tight')

def area_z_plot():
    x_plot = np.linspace(0, 1, 101)
    plt.plot(x_plot, area_n_z3, label='alpha_{}'.format(n))
    plt.legend()
    plt.grid()
    plt.xlabel("z")
    plt.ylabel("Area change")
    plt.show()

def plot():
    fig = plt.figure(dpi=200)
    # y = list(position_y(z))+list(position_y(z))+list(-position_y(z))+list(-position_y(z))
    # x = list(position_x(z))+list(-position_x(z))+list(position_x(z))+list(-position_x(z))
    
    
    y = list(position_y(z))+list(reversed(position_y(z)))
    x = list(position_x(z))+list(reversed(-position_x(z)))
    
    x2 = list(x) + list(reversed(x))
    y2 = list(y) + list(-position_y(z)) + list(reversed(-position_y(z)))
    
    
    yu = list(undeformed_y())+list(reversed(undeformed_y()))
    xu = list(undeformed_x())+list(reversed(-undeformed_x()))
    
    x2u = list(xu) + list(reversed(xu))
    y2u = list(yu) + list(-undeformed_y()) + list(reversed(-undeformed_y()))
    
    
    #print(x)
    # plt.plot(undeformed_x(),undeformed_y(),color='black',linewidth=0.75,linestyle=(0,(5,5)),label='undeformed cross section')
    # plt.plot(-undeformed_x(),-undeformed_y(),color='black',linewidth=0.75,linestyle=(0,(5,5)))
    # plt.plot(-undeformed_x(),undeformed_y(),color='black',linewidth=0.75,linestyle=(0,(5,5)))
    # plt.plot(undeformed_x(),-undeformed_y(),color='black',linewidth=0.75,linestyle=(0,(5,5)))
    
    # plt.plot(position_x(z),position_y(z),color='black',linewidth=0.75,label='deformed cross section')
    # plt.plot(-position_x(z),-position_y(z),color='black',linewidth=0.75)
    # plt.plot(-position_x(z),position_y(z),color='black',linewidth=0.75)
    # plt.plot(position_x(z),-position_y(z),color='black',linewidth=0.75)
    
    plt.plot(x2u,y2u,color='black',linewidth=0.75,linestyle='dashed',label='undeformed cross section')
    plt.plot(x2,y2,color='black',linewidth=0.75,label='deformed cross section')
    plt.legend()
    plt.axis([-1.5,1.5,-1,1])
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.title('Deformation of an initially cylindrical tube')
    #plt.grid()
    #fig.savefig("BB3.svg", bbox_inches='tight')
    #cmap, norm = mcolors.from_levels_and_colors([0, 2, 5, 6], ['red', 'green', 'blue'])
    #plt.scatter(x2, y2, c=x2, ec='k')
    plt.show()
    
def plot_sum(z):
    fig = plt.figure()
    # plt.plot(position_x_sum(z),position_y_sum(z),color='black',linewidth=0.75)
    # plt.plot(-position_x_sum(z),-position_y_sum(z),color='black',linewidth=0.75)
    # plt.plot(-position_x_sum(z),position_y_sum(z),color='black',linewidth=0.75)
    # plt.plot(position_x_sum(z),-position_y_sum(z),color='black',linewidth=0.75)
    
    # plt.plot(undeformed_x(),undeformed_y(),color='black',linewidth=0.75,linestyle='dashed')
    # plt.plot(-undeformed_x(),-undeformed_y(),color='black',linewidth=0.75,linestyle='dashed')
    # plt.plot(-undeformed_x(),undeformed_y(),color='black',linewidth=0.75,linestyle='dashed')
    # plt.plot(undeformed_x(),-undeformed_y(),color='black',linewidth=0.75,linestyle='dashed')
    
    
    y = list(position_y_sum(z))+list(reversed(position_y_sum(z)))
    x = list(position_x_sum(z))+list(reversed(-position_x_sum(z)))
    
    x2 = list(x) + list(reversed(x))
    y2 = list(y) + list(-position_y_sum(z)) + list(reversed(-position_y_sum(z)))
    
    
    yu = list(undeformed_y())+list(reversed(undeformed_y()))
    xu = list(undeformed_x())+list(reversed(-undeformed_x()))
    
    x2u = list(xu) + list(reversed(xu))
    y2u = list(yu) + list(-undeformed_y()) + list(reversed(-undeformed_y()))
    
    plt.plot(x2u,y2u,color='black',linewidth=1,linestyle='dashed')
    plt.plot(x2,y2,color='blue',linewidth=1)
    
    plt.axis([-1.5,1.5,-1,1])
    plt.xlabel('x')
    plt.ylabel('y')
    #fig.savefig("D9.svg", bbox_inches='tight')
    plt.show()

def eig_plot():
    plt.figure(dpi=200)
    plt.plot(t,Y_n,label='$\lambda_{}$='"{}".format(n,l_n))
    plt.legend()
    ax = plt.gca()
    ax.set_xticks(np.arange(0, np.pi/2 +0.01, np.pi/8))
    labels = ['$0$','$\pi/8$', r'$\pi/4$','$3\pi/8$', r'$\pi/2$']
    ax.set_xticklabels(labels)
    plt.show()

def eig_dx():
    plt.figure(dpi=200)
    plt.plot(t, Y_n, label="Y_{}".format(n))
    plt.plot(t,Y_np,label="Y'_{}".format(n))
    plt.legend()
    plt.show()

def y_position(z):
    plt.figure(dpi=200)
    plt.plot(t,position_y(z))
    plt.show()
    
def x_position(z):
    plt.figure(dpi=200)
    plt.plot(t,position_x(z))
    plt.show()

def mode_decay():
    fig = plt.figure()
    n3 = np.linspace(1,6,30)
    #plt.figure(dpi=200)
    #plt.xlabel("mode n")
    #plt.ylabel("$Q_n$")
    plt.scatter(n2,ll)
    plt.grid()
    #plt.plot(n3,abs(ll[0]*np.exp(1-n3)))
    #fig.savefig("D7.svg", bbox_inches='tight')
    plt.show()
    

def a_mode_decay():
    fig=plt.figure()
    #n3 = np.linspace(1,6,300)
    #plt.figure(dpi=200)
    plt.scatter(n2,ll2/A_Bar)
    #plt.plot(n3,abs(ll2[0]*np.exp(1-n3)))
    #plt.xlabel("n")
    #plt.ylabel("a_n")
    plt.grid()
    #fig.savefig("D8.svg", bbox_inches='tight')
    plt.show()
