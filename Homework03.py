import matplotlib.pyplot as plt
import numpy as np

#=========================================================================================

def euler(f1, f2, x0, t0, Tend, h):
    t = np.zeros(int(Tend/h)+1)
    x1 = np.zeros(len(t))
    x2 = np.zeros(len(t))
    x1[0] = x0; x2[0] = x0; t[0] = t0
    for n in range(0,len(t)-1):
        x1[n+1] = x1[n] + f1(x1[n],x2[n],t[n])*h
        x2[n+1] = x2[n] + f2(x1[n],x2[n],t[n])*h
        t[n+1] = t[n] + h
    plt.scatter(t,x1,s=10,color='red')
    plt.scatter(t,x2,s=10,color='blue')
    plt.xlabel('$t$')
    plt.ylabel('$x_1,x_2$')

#============================================================================================

def heun(f1, f2, x0, t0, Tend, h):
    t = np.zeros(int(Tend/h)+1)
    x1 = np.zeros(len(t))
    x2 = np.zeros(len(t))
    xa = np.zeros(len(t))
    xb = np.zeros(len(t))
    x1[0] = x0; xa[0] = x0; x2[0] = x0; xb[0] = x0; t[0] = t0
    for n in range(0,len(t)-1):
        t[n+1] = t[n] + h
        xa[n+1] = x1[n] + f1(x1[n],x2[n],t[n])*h
        xb[n+1] = x2[n] + f2(x1[n],x2[n],t[n])*h
        x1[n+1] = x1[n] + h/2 *(f1(x1[n],x2[n],t[n])+f1(xa[n+1],xb[n+1],t[n+1]))
        x2[n+1] = x2[n] + h/2 *(f2(x1[n],x2[n],t[n])+f2(xa[n+1],xb[n+1],t[n+1]))
    plt.scatter(t,x1,s=10,color='orange')
    plt.scatter(t,x2,s=10,color='green')
    plt.xlabel('$t$')
    plt.ylabel('$x_1,x_2$')

#=============================================================================================

def rk4(f1, f2, x0, t0, Tend, h):
    t = np.zeros(int(Tend/h)+1)
    x1 = np.zeros(len(t))
    x2 = np.zeros(len(t))
    x1[0] = x0; x2[0] = x0; t[0] = t0
    for n in range(0,len(t)-1):
        t[n+1] = t[n] + h
        k1 = f1(x1[n], x2[n], t[n])
        c1 = f2(x1[n], x2[n], t[n])
        k2 = f1(x1[n] + h * k1 / 2, x2[n] + h * c1 / 2, t[n] + h / 2)
        c2 = f2(x1[n] + h * k1 / 2, x2[n] + h * c1 / 2, t[n] + h / 2)
        k3 = f1(x1[n] + h * k2 / 2, x2[n] + h * c2 / 2, t[n] + h / 2)
        c3 = f2(x1[n] + h * k2 / 2, x2[n] + h * c2 / 2, t[n] + h / 2)
        k4 = f1(x1[n] + h * k3, x2[n] + h * c3, t[n] + h)
        c4 = f2(x1[n] + h * k3, x2[n] + h * c3, t[n] + h)
        x1[n+1] = x1[n] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x2[n+1] = x2[n] + h * (c1 + 2 * c2 + 2 * c3 + c4) / 6
    plt.scatter(t,x1,s=10,color='purple')
    plt.scatter(t,x2,s=10,color='deeppink')
    plt.xlabel('$t$')
    plt.ylabel('$x_1,x_2$')

#==============================================================================================

def opinion_function_agent1(x1,x2,t):
    global d; u; a; j; b
    return (-d*x1 + u*np.tanh(a*x1+j*x2) + b)

def opinion_function_agent2(x1,x2,t):
    global d; u; a; j; b
    return (-d*x2 + u*np.tanh(a*x2+j*x1) + b)

d=1; u=0.31; a=1.2; j=-1.3; b=0

#===============================================================================================

plt.figure(1)
euler_plot = euler(opinion_function_agent1, opinion_function_agent2,-1,0,7,5e-1)
#euler_plot = euler(opinion_function_agent1, opinion_function_agent2,-1,0,7,1e-3) # b)
heun_plot = heun(opinion_function_agent1, opinion_function_agent2,-1,0,7,5e-1)
rk4_plot = rk4(opinion_function_agent1, opinion_function_agent2,-1,0,7,5e-1)
plt.show()