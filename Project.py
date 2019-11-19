import scipy as sp
import pylab as plt
import numpy as np
import math
import random
from scipy.integrate import odeint

#https://hodgkin-huxley-tutorial.readthedocs.io/en/latest/_static/Hodgkin%20Huxley.html

class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""
    sick = 0
    array = []
    C_m  = 0
    g_Na = 0
    g_K  = 0
    g_L  = 0
    E_Na = 0
    E_K  = 0
    E_L  = 0
    t = sp.arange(0.0, 500, 0.001)

    def __init__(self, sicks): 
        #print(sicks)
        self.sick = sicks
        healthy = [10.0, 1.2, 0.36, 0.003, 50.0, -77.0, -54.387]
        #calculate using modified nernst and log rules
        # 200% increase in sodium --> 69.9%
        # 8-15% increase in potassium --> 10% decrease
        # C_m, g_Na, g_K, g_L, E_Na, E_K, E_L
        alzheimers = [10.0, 1.2, 0.36, 0.003, 84.5, -77.7, -54.387]
        #print(sicks)
        if sicks:
            self.array = alzheimers
        else: 
            self.array = healthy

        self.C_m  = self.array[0]
        self.C_m = self.C_m/10
        """membrane capacitance, in uF/cm^2"""

        self.g_Na = self.array[1]
        self.g_Na = self.g_Na*100
        """Sodium (Na) maximum conductances, in mS/cm^2"""

        self.g_K  =  self.array[2]
        self.g_K  =  self.g_K*100
        """Postassium (K) maximum conductances, in mS/cm^2"""

        self.g_L  =   self.array[3]
        self.g_L  =   self.g_L*100
        """Leak maximum conductances, in mS/cm^2"""

        self.E_Na =  self.array[4]
        """Sodium (Na) Nernst reversal potentials, in mV"""

        self.E_K  = self.array[5]
        """Postassium (K) Nernst reversal potentials, in mV"""

        self.E_L  = self.array[6]
        """Leak Nernst reversal potentials, in mV"""
        self.t = sp.arange(0.0, 500, 0.001)
    

    def alpha_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.1*(V+40.0)/(1.0 - sp.exp(-(V+40.0) / 10.0))

    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4.0*sp.exp(-(V+65.0) / 18.0)

    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*sp.exp(-(V+65.0) / 20.0)

    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))

    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(V+55.0)/(1.0 - sp.exp(-(V+55.0) / 10.0))

    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*sp.exp(-(V+65) / 80.0)

    def I_Na(self, V, m, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)

        |  :param V:
        |  :param m:
        |  :param h:
        |  :return:
        """
        return self.g_Na * m**3 * h * (V - self.E_Na)

    def I_K(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_K  * n**4 * (V - self.E_K)
    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_L * (V - self.E_L)

    def I_inj(self, t):
        """
        External Current

        |  :param t: time
        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400
        """
        #ie_A = 300
        #ie_A = 200
        #return ie_A/10
        return 10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>400)

    @staticmethod
    def dALLdt(X, t, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        V, m, h, n = X
        if self.C_m == 10.0:
            print("damn")

        dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m

        dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        
        return dVdt, dmdt, dhdt, dndt

    def Main(self):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        if self.sick == 0:
            X = odeint(self.dALLdt, [-65, 0.0529, 0.5961, 0.3177], self.t, args=(self,))
        else:
            X = odeint(self.dALLdt, [-65, 0.0529, 0.9, 0.3177], self.t, args=(self,))
    
        V = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)
        
        i = 0
        maxi = len(V)
        spike = 0
        up = 1
        spikescore = 0
        while (i<maxi):
            if up == 1:
                if V[i-1]>V[i]:
                    if V[i]>0:
                        spike = spike + 1 
                        spikescore = spikescore + V[i-1]
                    up = 0
            if up == 0:
                if V[i]>V[i-1]:
                    up = 1
            i = i + 1
        # print(spike)
        print(spikescore)
        # print(spikescore/spike)
        return(spikescore)



if __name__ == '__main__':
    alzheimers = HodgkinHuxley(1)
    healthy = HodgkinHuxley(0)

    sick = alzheimers.Main()
    heal = healthy.Main()
    
    print("asdfadf\n\n")
    print(sick)
    print(heal)

    
    w = np.random.rand(10,10)
    wt = w
    w0 = np.random.rand(10,10)
    w0t = w0
    t = np.random.rand(10,10)
    tt = t
    i = 0
    xaxis = []
    yaxis = []

    datastorage = []
    while i<1000:
        x = np.random.rand(10,1)
        datastorage.append(x)
        y = np.matmul(t,x)
        xaxis.append(i)
        h = np.matmul(w0,x)
        y_n = np.matmul(w,h)

        error = np.subtract(y,y_n)

        delta_w = -np.matmul(error,h.T)*0.01
        w = np.subtract(w,delta_w)

        tempy = np.matmul(error,x.T)
        delta_w0 = -(np.matmul(w.T,tempy))*0.01
        w0 = np.subtract(w0,delta_w0)

        yaxis.append(np.linalg.norm(error))

        i = i + 1

    w = wt
    w0 = w0t
    t = tt
    i = 0
    xaxis1 = []
    yaxis1 = []
    newaxis = []
    tempaxis = []
    sickness = 1
    while i<1000:
        x = datastorage[i] #ensuring that both are using the same training data
        y = np.matmul(t,x)
        xaxis1.append(i)
        h = np.matmul(w0,x)
        y_n = np.matmul(w,h)
        error = np.subtract(y,y_n)

        #sickness = bool(random.getrandbits(1))
        sickness = 1
        if sickness:
            error = error*sick

        delta_w = -np.matmul(error,h.T)*0.01
        w = np.subtract(w,delta_w)
        tempy = np.matmul(error,x.T)
        delta_w0 = -(np.matmul(w.T,tempy))*0.01
        w0 = np.subtract(w0,delta_w0)
        yaxis1.append(np.linalg.norm(error))
        i = i + 1


    plt.figure()
    plt.subplot(3,1,1)
    plt.title('error 1')
    plt.plot(xaxis, yaxis, 'k')
    plt.ylim(0,5)
    plt.ylabel('Error')


    plt.subplot(3,1,3)
    plt.title('error 2')
    plt.plot(xaxis1, yaxis1, 'k')
    plt.ylim(0,5)
    plt.ylabel('Error')


    plt.xlabel('Iteration')
    plt.savefig('plots.png')
    plt.show()


















