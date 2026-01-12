# Backpropagation

[Micrograd](https://github.com/karpathy/micrograd) is an implementatino for Auto Gradiant.


[Research pater introduced this ackpopegation - Learnign representations by back propagating errors](https://gwern.net/doc/ai/nn/1986-rumelhart-2.pdf)

What is backpropagation

Alog to compute the Gradient (What is this gradient) ??

Negative gradient of the cost funtion. -> whihc tell how th3 weight and baiases are cto be changed and the biases to efficeintely decrease the cost. 


Terms to understand

- Weights
- Bias
- Activation
- Gradiant
- Negative Gradient
- Stochastic gradiat decent. 




In the context of the provided sources, **backpropagation** is the core algorithm that enables neural networks to learn by determining how to most efficiently adjust their internal settings to reduce errors,. It is the mathematical tool used to compute the "crazy complicated" **gradient** of a cost function.

Here is an intuitive breakdown of the process:

### 1. Finding the "Nudges"
Backpropagation starts by looking at a single training example and identifying how the output layer's activations should be "nudged" to reach the correct answer,. For instance, if the network is shown a handwritten "2," backpropagation calculates how much the activation for the "2" neuron should increase and how much the other neurons should decrease. These nudges are **proportional** to how far away each value is from its target.

### 2. Assessing Sensitivity and Influence
The algorithm determines which weights and biases are most responsible for the current output. This is often described as the **sensitivity** of the cost function to each parameter.
*   **Weights and Activations:** Connections with "brighter" (more active) neurons in the previous layer have a stronger influence on the final result.
*   **Bang for Your Buck:** Backpropagation focuses on the changes that provide the most significant decrease in cost for the effort expended.

### 3. Propagating Backwards Recursively
Because we cannot change the activations of neurons directly—only the weights and biases that create them—the algorithm calculates what changes *should* happen in the previous layer to satisfy the needs of the current layer. 
*   Each neuron in the output layer has its own "desires" for what the previous layer should look like.
*   Backpropagation **adds these desires together** for every neuron in the layer and then recursively applies the same logic to the next layer back.
*   This process continues until the algorithm has moved through the entire network from the output back to the input.

### 4. Averaging and Efficiency
A single training example only shows how the network should change for that specific case; listening only to one example would cause the network to ignore all others. Therefore, the algorithm performs this routine for thousands of examples and **averages the desired changes** to find the overall negative gradient,. 

To speed this up, researchers use **mini-batches**—randomly shuffling data and calculating updates for small groups (e.g., 100 examples) at a time—a technique known as **stochastic gradient descent**,.

***

**Analogy for Stochastic Gradient Descent**
The sources describe the movement toward a better model using an analogy: instead of being a "carefully calculating man" who spends a long time determining the exact downhill direction for one slow, perfect step, the algorithm acts more like a **"drunk man stumbling aimlessly down a hill"** who takes many quick, slightly imprecise steps that eventually lead to the bottom much faster.
