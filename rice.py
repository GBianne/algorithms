import numpy as np
import random
import scipy.optimize as opt
import scipy.integrate as integrate
import matplotlib.pyplot as plt

#/!\ only works with positive values (right tail) - if working with negative values, abs() them

x0 = []

x0 = [ 3.58, 3.6,3.62,3.63,3.635,3.635,3.7,3.71,3.72,3.73,3.74, 3.75, 3.78,3.79,3.8,3.82,3.86,3.86,3.86,3.86,3.87,3.87,3.88,3.89,3.89,3.9,3.9,3.91,3.93,3.94,3.96,3.96,3.96,3.97,3.98,3.99,4.0,4.,4.,4.03,4.06,4.08,4.08,4.09,4.1,4.12,4.12,4.12,4.18,4.18,4.2,4.2,4.22,4.24,4.24,4.26,4.32,4.32,4.36,4.37,4.55,4.55,4.7 ]

#with open("/media/sf_shared_vm/Ulm/extrapolation/trafficSimulation_GTBulm_blockMaxima_mid.csv",'r') as ulm:
#with open("/media/sf_shared_vm/Ulm/extrapolation/trafficSimulation_GTBulm_blockMaxima_third.csv",'r') as ulm:
# with open("/media/sf_shared_vm/Ulm/extrapolation/trafficSimulation_GTBulm_blockMaxima_tie.csv",'r') as ulm:
#     contents = ulm.read().splitlines()
#     for line in contents[1:]:
#         x0.append(float(line))

xmin = min(x0)
xmax = max(x0)

random.shuffle(x0)
print(x0)

#STEP 1 - level crossings

n_classes = 10
step = (xmax-xmin)/n_classes
print(xmin, xmax, step)
#classes: ]xmin;...] ... ]...;xmax]
#nb: xmax/min are not the real xmax/min, but rather the values calculated over the first blank run
#]-inf;xmin] and ]xmax;+inf[ are additional classes to account for it
#to add in C++

levels = []
for k in range(n_classes):
    levels.append(xmin + k*step)
    
crossings = [0 for k in range(n_classes)]
    
current_class = -1

for x in x0:
    #1 - determine effect class
    if x == xmax:
        effect_class = n_classes-1
    else:
        effect_class = int((x-xmin)/step)
    #2 - compare with current level
    if current_class == -1:
        #not initialized
        current_class = effect_class
    elif effect_class > current_class:
        #one or more levels have been crossed
        for i in range(current_class+1, effect_class+1):
            crossings[i] += 1
        current_class = effect_class
    elif effect_class < current_class:
        #going back to a lower level
        current_class = effect_class
print(crossings)
print(levels)

#STEP 2 - Rice's formula
#for each xd:
#1 - least squares
#2 - density functions
#3 - cumulative distributions
#4 - KS test statistics value
#5 - Qks value
#once xd = xpeak, get the optimal value xopt = max(xd) | Qks >= b, where b is a set threshold within [0.9;1]
beta0 = 0.99
xpeak = 0
for k in range(len(crossings)):
    if crossings[k] == max(crossings):
        xpeak = k

#least squares adjustment function
def lnv_adj(x, v0, m, sig):
    return np.log(v0) - m**2/(2*sig**2) + x*m/sig**2 - x**2/(2*sig**2)

#adjusted density
def vd_adj(x, v0, m, sig, d):
    if x < levels[d]:
        return 0
    else:
        return step*np.exp(lnv_adj(x, v0, m, sig))/integrate.quad(lambda u: np.exp(lnv_adj(u, v0, m, sig)), levels[d], np.inf)[0]

#real density
def vd(x, crossings, levels, d):
    if x < levels[d]:
        return 0
    if x > xmax+step:
        return 0
    #step function
    res = 0.
    step_integral = 0.
    for k in range(d, len(levels)):
        step_integral += crossings[k]
        if x >= levels[k]:
            res = crossings[k]
    return res/step_integral

#adjusted cumulative distribution
def Fd_adj(x, v0, m, sig, d):
    return integrate.quad(lambda u: vd_adj(u, v0, m, sig, d), -np.inf, x)[0]/step

#real cumulative distribution
def Fd(x, crossings, levels, d):
    res = 0.
    for k in range(d, len(crossings)):
        if x >= levels[k]:
            if x >= levels[k]+step:
                res += vd(levels[k], crossings, levels, d)
            else:
                res += vd(levels[k], crossings, levels, d) * (x-levels[k])/step
    return res

#KS test
def deviation(x, v0, m, sig, d, crossings, levels):
    return -abs(Fd_adj(x, v0, m, sig, d) - Fd(x, crossings, levels, d))

def Dxd(v0, m, sig, d, crossings, levels, precision):
    #manual minimization of the deviation
    #both cumulative distributions match on ]-inf;xmin[ U ]xmax;+inf[
    x_range = np.linspace(xmin, xmax,int(1./precision))
    max_dev = 0
    x_max = 0
    for x in x_range:
        dev = abs(Fd_adj(x, v0, m, sig, d) - Fd(x, crossings, levels, d))
        if dev > max_dev:
            max_dev = dev
            x_max = x
    return (x_max, max_dev)

def Bxd(d, dxd, n):
    res = 0
    for k in range(1,n):
        res += 2*(-1)**(k-1)*np.exp(-2*k**2*(len(crossings)-d)*dxd**2)
    return res

#test functions
def plot_crossings(crossings, levels, d, v0, m, sig, n):
    x_data = np.linspace(xmin-5*step, xmax+5*step, n)
    x_adj_data = np.linspace(levels[d]-step/10.,xmax+5*step,n)
    y_data = [crossings[int((x-xmin)/step)] if (x>=xmin and x<xmax) else 0 for x in x_data]
    y_adj_data = [np.exp(lnv_adj(x,v0,m,sig)) for x in x_adj_data]
    plt.plot(x_data, y_data, color='red')
    plt.plot(x_adj_data, y_adj_data, color='blue')
    plt.title("Level crossings")
    plt.show()

def plot_densities(crossings, levels, d, v0, m, sig, n):
    x_data = np.linspace(xmin-5*step, xmax+5*step, n)
    y_data = [vd(x, crossings, levels, d) for x in x_data]
    y_adj_data = [vd_adj(x, v0, m, sig, d) for x in x_data]
    plt.plot(x_data, y_data, color='red')
    plt.plot(x_data, y_adj_data, color='blue')
    plt.title("Densities")
    plt.show()

def plot_cumulative(crossings, levels, d, v0, m, sig, n):
    x_data = np.linspace(levels[d]-step, xmax+5*step, n)
    y_data = [Fd(x, crossings, levels, d) for x in x_data]
    y_adj_data = [Fd_adj(x, v0, m, sig, d) for x in x_data]
    plt.plot(x_data, y_data, color='red')
    plt.plot(x_data, y_adj_data, color='blue')
    plt.title("Cumulative distributions")
    plt.show()

#loop on xd
#first value: only three classes taken into account
#last value: all classes starting from xpeak
dopt = 0
params_opt = [0,0,0]
for d in range(len(crossings)-3, xpeak-1, -1):
    #step 2.1 - adjust least squares
    data_x = []
    data_y = []
    for i in range(d,len(crossings)):
        data_x.append(levels[i])
        data_y.append(np.log(crossings[i]))
    x0 = [1,1,1]
    print()
    print(data_x, data_y)
    params = opt.curve_fit(lnv_adj, data_x, data_y, x0, bounds=([0,-np.inf,0],[np.inf,np.inf,np.inf]), maxfev=100000)[0]
    v0_opt = params[0]
    m_opt = params[1]
    sig_opt = params[2]
    print(params)
    
    for x in data_x:
        print(np.exp(lnv_adj(x, v0_opt, m_opt, sig_opt)), end = " ")
    print()
    for y in data_y:
        print(np.exp(y), end = " ")
    print()
    print("\n")
    
    dxd_val = Dxd(v0_opt, m_opt, sig_opt, d, crossings, levels, 1e-2)
    print("Maximum deviation:", dxd_val)
    beta = Bxd(d, dxd_val[1], 1000)
    print ("Kolmogorov-Smirnov test value:", beta)
    
    #plot_crossings(crossings, levels, d, v0_opt, m_opt, sig_opt, 1000)
    #plot_densities(crossings, levels, d, v0_opt, m_opt, sig_opt, 1000)
    #plot_cumulative(crossings, levels, d, v0_opt, m_opt, sig_opt, 20)
    
    if beta > beta0:
        dopt = d
        params_opt = [v0_opt, m_opt, sig_opt]

print("Optimal class for Rice's formula adjustment:", dopt)
print("Optimal parameters [v0, m, sigma]:", params_opt)
v0_opt = params_opt[0]
m_opt = params_opt[1]
sig_opt = params_opt[2]

def rvalue(r,t):
    return m_opt+sig_opt*np.sqrt(2*np.log(v0_opt*r/t))

tref = 500. #500-day data

print("Return value at r=1 year", rvalue(365.25,tref))
print("Return value at r=10 years", rvalue(365.25*10,tref))
print("Return value at r=100 years", rvalue(365.25*100,tref))
print("Return value at r=1000 years", rvalue(365.25*1000,tref))
print("Return value at r=10000 years", rvalue(365.25*10000,tref))
print("Return value at r=100000 years", rvalue(365.25*100000,tref))
print("Return value at r=1000000 years", rvalue(365.25*1000000,tref))