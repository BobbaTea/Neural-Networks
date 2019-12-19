# Neural-Networks

## Background
I completed this project in my freshman year (two years ago) and at the time started with the goal of achieving fluency with deep learning basics, and with feed forward neural networks. I have recently continued working and revising to refine the efficiency of the code. Some of the changes have been to port the original code from c++ to Java.

## What is it
This is a feed-forward neural network utilizing back-propagation (gradient descent) as its primary learning algorithm. Written from scratch it doesn't make use of any open-source code or third party resources(TensorFlow, Keras, etc). 
It is a repurposable neural network that can be scaled to derive patterns and over time make more accurate predictions. A good example of this (and what the main tester method accomplishes) is to learn how to add simple values together. From its starting initialized point the network predicts seemingly random values. Over multiple iterations of learning (through back-propagation) it refines specific weights and biases which define the network. This in essence, allows it to self-refine its ability to add. 

## How does it work
The fundamentals of this neural network boils down to using dot products to forward propagate the neural network. The error/cost function (difference from expected result) is then calculated and partial derivatives of the cost to various weights and biases throughout the net are used to tweak these variables. 
View the output.txt file to see an example of the outputted data. The "Time" attribute gives the runtime of the forward and back propagation methods. It is followed by the number of iterations of learning and propagating. Then the inputs are displayed along with the expected output. Finally the actual output from the network and the error are shown. 
It is evident that overtime (over iterations) the error rate value is decreasing and the predicted answers are moving from random and sporatic to closer to the expected value.
