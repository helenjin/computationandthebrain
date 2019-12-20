# Computation and the Brain

Professor Christos H. Papadimitriou

Fall 2019

Helen Jin and Yi Jun Lim

------
#### Final Draft of Project Update (12/1/2019)
An important task of the brain is to identify and differentiate different visual patterns. So, when the brain deteriorates due to atrophy and/or other malfunction from Alzheimer’s Disease, we expect to see a worse performance than usual. 
In order to test this out, we examined a basic classification task for both the healthy and unhealthy neurons to implement. Two distinct patterns of a cross and circle are presented to each respective network, and each image is presented for 200 ms with a maximum spike frequency of 50 Hz. As time progresses, the weights align themselves according to the input. For our experiments, we chose to use the Hodgkin-Huxley model for our neurons, with Alzheimer’s-afflicted sick neurons represented by the parameters we previously derived from the literature above. So using the new values with unhealthy ion concentrations, we adjusted Brian to run the learning algorithm, the STDP rule. As time progresses, the weights continuously align themselves according to the input. 
We allowed the SSN to learn the cross or circle pattern for 20, 50, 100 and 200 ms. Observing the results, it seems that the healthy SNN learns at a slightly faster rate. For example, it is clear that by 50ms, the SNN composed of healthy neurons already learns the cross pattern, while the SNN composed of unhealthy neurons has yet to. However, the difference is not too significantly noticeable. At the end of the 200 ms, both the healthy and unhealthy SNNs are able to learn both patterns relatively well.
For next steps, one possibility is that we can extend this classification task to include other, preferably more complex and difficult images to classify, such as the MNIST dataset. We would still expect the healthy neurons to perform better, but it will be interesting to see how the process differentiates from the original more basic task. These results will help develop a more refined understanding of how exactly each neural network of either healthy or unhealthy Alzheimer’s afflicted neurons function and go about learning. Also, perhaps a particular architecture, possibly with more layers and given more time to train, of the neural network could perform better. Due to limited time, we were unable to complete the aforementioned extensions, but they are good places to start to continue this work in the future. 
Furthermore, the effects of having the affected ion concentrations do not seem to have much of an effect on the SNN’s ability to learn. Whilst there was a slight difference, it was not drastic enough. This seems to be in line with new literature that states that targeting ion channels created by Aβ in people with AD may not be the best treatment plan. We did not know this before we started the experiment, but as we continued to perform research for it, we stumbled upon some articles that stated that they are trying to look into other treatments for AD. Therefore, it seems that this project would benefit from more research of AD. We need to gain a better understanding of the actual biological condition of AD first before we can better computationally model it. However, rather than it just being a one way street, this is a two-way street where both biology (or perhaps more appropriately, the brain) and computation enrich each other to a better understanding of the other. Thus, there is indeed much merit to exploring this intersection of the brain and computation.


#### First Draft of Project Update (11/19/2019)

This week, we did some research on the parameters representing the healthy and 
and updated the parameters accordingly so. We took the various ion concentrations and recalculated how (we believe) it would change the different volatage values in the hodgkin huxley model. We then also did research on how the behaviour of neurons in Alzherimer's patients may affect the gating probabilities, and applied it to the Hodgkin Huxley model. In the results of modelling the outputs of the neurons, we can see that based on our updated parameters, a neuron with Alzheimer's-like parameters tend to be overly active, and given the same input, there is more spikes, and also a higher difference in magnitude. This is in line with the literature that we found, that showed that in diseased mice brain slices that were studied, neurons that were diseased were more exitable. Based on our preliminary findings, we believe that this is a significant enough difference that it might affect the SNN. However, we have yet to implement an SNN. We also explored ways we can continue to use object oriented programming, which we are quite rusty in, to create neuron objects that take in parameters and output values, so we can use it in our modelling of the SNN. Our next steps for the coming week is to finish a simple SNN and train it to see if our hypothesis that the diseased neuron will have more difficulty learning, holds. And if hypothesis doesn't hold, why.


Initial steps building down the foundation of the Hodgkin-Huxley model for a neuron, healthy and Alzheimer diseased, were taken in the Project.py file. 

To build the Spiking Neural Network (SNN), we will incorporate many of these Hodgkin-Huxley models for neurons in the structure consisting of two (and later three) hidden layers. To train the SNN, we have decided to try SpikeProp first. SpikeProp assumes that each neuron in the network fires exactly once during the simulation interval. 

![A sequential step](spikeprop.jpg?raw=true "Spike Propagation")


#### Citations
<https://doi.org/10.1016/j.neunet.2016.10.011>

Please note that our code was heavily influenced/based on: 





