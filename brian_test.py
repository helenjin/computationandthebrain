from common import *

x,y = read_dataset()

#normalizing data
normalized = (x-np.min(x))/float(np.max(x)-np.min(x))*50 #50 Hz is the maximum firing rate

#stdp parameters
sigma = 0.3                                                                                                                                                                                                             
taupre = 5*ms 
taupost = 8*ms
wmax = 1.2
wmin = -0.5
Apre = 0.4*sigma
Apost = -0.2*sigma

#characteristics of neurons
num_neurons = 1
duration = 2*second


sick = 1
runtime = 200


## Parameters 
area = 20000*umetre**2
Cm = 1*ufarad*cm**-2 * area
gl = 5e-5*siemens*cm**-2 * area
El = -65*mV
EK = -77*mV
ENa = 50*mV
g_na = 120*msiemens*cm**-2 * area
g_kd = 36*msiemens*cm**-2 * area
VT = -63*mV

if sick == 1:
      EK = -78.98
      ENa = 21.82*mV
# EK = EK
# ENa = ENa




# equations governing the dynamics of neurons --> according to Hodgkin-Huxley model
eqs = Equations('''
dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
    (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
    (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
    (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
I : amp
''')


# initializing weights
S_initial = np.random.uniform(low=0,high=0.3,size=(par.hid_size,par.vis_size))

for i in range(par.epochs):

      for j in range(len(x)):

            #output layer neuron
            # Threshold and refractoriness are only used for spike counting
            G = NeuronGroup(num_neurons, eqs,
                    threshold='v > -50*mV',
                    refractory='v > -50*mV',
                    method='exponential_euler')
            #input neurons firing according to Poisson distribution with rates determined by the intensity of the corresponding input pixel
            P = PoissonGroup(par.vis_size, x[j]*Hz)

            #synapse governed by the rules of STDP
            S = Synapses(P, G,
                         '''
                         w : 1
                         dapre/dt = -apre/taupre : 1 (event-driven)
                         dapost/dt = -apost/taupost : 1 (event-driven)
                         ''',
                         on_pre='''
                         v_post += w*mV*2
                         apre += Apre
                         w = clip(w+apost, wmin, wmax)
                         ''',
                         on_post='''
                         apost += Apost
                         w = clip(w+apre, wmin, wmax)
                         ''')

            S.connect()
            S.w = S_initial[j]

            M = StateMonitor(G, 'v', record=True) #monitors the membrane voltage
            spikemon = SpikeMonitor(P) #records spikes from input neurons
            spikemon1 = SpikeMonitor(G) #records spikes from output neurons
            run(runtime*ms)

            #updating the weights
            S_initial[j] = S.w


#reconstructing weights to analyse the training
recon_weights(S_initial)