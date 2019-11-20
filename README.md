# Computation and the Brain

Professor Christos H. Papadimitriou

Fall 2019

Helen Jin and Yi Jun Lim

------

#### First Draft of Project Update (11/19/2019)

This week, we did some research on the parameters representing the healthy and 
and updated the parameters accordingly so. We took the various ion concentrations and recalculated how (we believe) it would change the different volatage values in the hodgkin huxley model. We then also did research on how the behaviour of neurons in Alzherimer's patients may affect the gating probabilities, and applied it to the Hodgkin Huxley model. In the results of modelling the outputs of the neurons, we can see that based on our updated parameters, a neuron with Alzheimer's-like parameters tend to be overly active, and given the same input, there is more spikes, and also a higher difference in magnitude. This is in line with the literature that we found, that showed that in diseased mice brain slices that were studied, neurons that were diseased were more exitable. Based on our preliminary findings, we believe that this is a significant enough difference that it might affect the SNN. However, we have yet to implement an SNN. We also explored ways we can continue to use object oriented programming, which we are quite rusty in, to create neuron objects that take in parameters and output values, so we can use it in our modelling of the SNN. Our next steps for the coming week is to finish a simple SNN and train it to see if our hypothesis that the diseased neuron will have more difficulty learning, holds. And if hypothesis doesn't hold, why.


Initial steps building down the foundation of the Hodgkin-Huxley model for a neuron, healthy and Alzheimer diseased, were taken in the Project.py file. 

To build the Spiking Neural Network (SNN), we will incorporate many of these Hodgkin-Huxley models for neurons in the structure consisting of two (and later three) hidden layers. To train the SNN, we have decided to try SpikeProp first. SpikeProp assumes that each neuron in the network fires exactly once during the simulation interval. 

![A sequential step](spikeprop.jpg?raw=true "Spike Propagation")


#### Citations
<https://doi.org/10.1016/j.neunet.2016.10.011>



