

/*



*/


package feedforward;

import java.text.DecimalFormat;
import java.util.ArrayList;

//class representing a neural net and all its corresponding layers and neurons
public class ffNet {

	// global variables used in recursion method
	ArrayList<Integer> counter;
	ArrayList<layer> network;
	double total = 1;

	// layer dimensions
	final int dimen[] = { 2, 3, 1 };
	// learning rate (used to determine step size)
	// larger rates can overstep the global minimum
	// smaller rates can significantly reduce learning time
	final double lrate = 0.1;

	int numLayers = dimen.length;

	public ffNet() {
		counter = new ArrayList<Integer>();
		network = new ArrayList<layer>();
		for (int i = 0; i < dimen.length; i++) {
			if (i != dimen.length - 1) {
				network.add(new layer(dimen[i], dimen[i + 1]));
			} else {
				network.add(new layer(dimen[i], 0));
			}
		}
	}

	// Forward propagate/feed forward method
	void forwardPropagate(double input[]) {
		double newval = 0;
		// Set input array values to input neuron's 'activation' variable
		// Iterate through neurons
		for (int m = 0; m < dimen[0]; m++) {
			network.get(0).layer.get(m).activation = input[m];
		}
		// Run feedforward through all layers using values and weights
		// Iterate through layers
		for (int i = 1; i < dimen.length; i++) {
			// Iterate through neurons in current layer
			for (int m = 0; m < dimen[i]; m++) {
				newval = 0.0;
				// Iterate through neurons in previous layer for weights and values
				for (int a = 0; a < dimen[i - 1]; a++) {
					newval = newval + (network.get(i - 1).layer.get(a).activation
							* network.get(i - 1).layer.get(a).weights.get(m));
				}
				newval = newval + (network.get(i).layer.get(m).bias);

				// Save newval which is currently the net inside the n variable
				network.get(i).layer.get(m).net = newval;

				// Optional: Apply Sigmoid or ReLU with bias on val of current layer neuron
				// network.get(i).layer.get(m).activation = 1 / (1 + Math.pow(Math.E, -newval));

				// tanh activation function
				network.get(i).layer.get(m).activation = Math.tanh(newval);

			}
		}
	}

	// Backpropagation method
	void backPropagate(double opair[]) {

		// Iterate through layers
		for (int i = 0; i < numLayers; i++) {
			// Iterate through Neurons
			for (int m = 0; m < dimen[i]; m++) {
				total = 0;
				// For the last layer
				if (i == numLayers - 1) {
					multPaths(i, m, opair, true, numLayers - 1, -1);
					network.get(i).layer.get(m).bias -= lrate * total;

					// For the first layer
				} else if (i == 0) {
					// Weights
					for (int l = 0; l < dimen[i + 1]; l++) {
						multPaths(i, m, opair, false, numLayers - 1, l);

						network.get(i).layer.get(m).weights.set(l,
								network.get(i).layer.get(m).weights.get(l) - (lrate * total));
					}

					// All hidden layers
				} else {
					// Biases
					multPaths(i, m, opair, true, numLayers - 1, -1);
					network.get(i).layer.get(m).bias -= lrate * total;
					// Weights
					for (int l = 0; l < dimen[i + 1]; l++) {
						multPaths(i, m, opair, false, numLayers - 1, l);
						network.get(i).layer.get(m).weights.set(l,
								network.get(i).layer.get(m).weights.get(l) - (lrate * total));
					}

				}
			}
		}
	}

	// Recursive method for using partial derivatives to adjust weights and biases
	// Solely called by the backPropagate function
	void multPaths(int aLayer, int aNeuron, double[] opair, boolean type, int current, int weightIndex) {
		// If this is the first iteration reset all the variables as they are global
		if (current == numLayers - 1) {
			counter.clear();
			for (int s = 0; s < numLayers; s++) {
				if (s == aLayer) {
					counter.add(aNeuron);
				} else {
					counter.add(0);
				}
			}
		}

		// If the current layer is the target layer to stop at
		if (current == aLayer) {
			if ((type == false && counter.get(aLayer + 1) == weightIndex) || type == true) {
				double val = 1;
				for (int x = counter.size() - 1; x >= aLayer; x--) {
					if (x == counter.size() - 1) {
						// Last layer
						val = dCost(network.get(numLayers - 1).layer.get(counter.get(numLayers - 1)).activation,
								opair[counter.get(numLayers - 1)])
								* dSigmoid(network.get(numLayers - 1).layer.get(counter.get(numLayers - 1)).net);
					}
					if (x == aLayer) {
						// Adjustment layer
						if (type == false) {
							val *= dNetWeight(x, counter.get(x));
						}
						total += val;
						break;
					}
					if (x != aLayer && x != counter.size() - 1) {
						// All other layers
						val *= dNetActivation(x, counter.get(x), counter.get(x + 1));
						val *= dSigmoid(network.get(x).layer.get(counter.get(x)).net);
					}
				}
			}
		} else {
			for (int i = 0; i < dimen[current]; i++) {
				counter.set(current, i);
				multPaths(aLayer, aNeuron, opair, type, current - 1, weightIndex);
			}
		}
	}

	// Derivative functions
	public double dSigmoid(double x) {
		// sigmoid Activation
		// return (Math.pow(Math.E, -x)) / (Math.pow((1 + Math.pow(Math.E, -x)), 2));

		// tanh Activation (faster learning rate empirically through tests)
		return 1 - Math.pow(Math.tanh(x), 2);

	}

	// derivative of cost function
	public double dCost(double activation, double target) {
		return 2 * (activation - target);
	}

	// derivative of Bias with respect to Net
	public double dNetBias() {
		return 1;
	}

	// derivative of Weight with respect to Net
	public double dNetWeight(int l, int k) {
		return network.get(l).layer.get(k).activation;
	}

	// derivative of activation of neuron in layer L-1 with respect to Net
	public double dNetActivation(int l, int j, int k) {
		return network.get(l).layer.get(j).weights.get(k);

	}

	// chain rule of the dCost and dSigmoid
	public double dCostNet(double activation, double target, double net) {
		return dCost(activation, target) * dSigmoid(net);
	}

	// MAIN!!!!
	public static void main(String args[]) {

		// create ffNet object
		ffNet n = new ffNet();
		System.out.println(n.numLayers);

		// variables for input and output
		double a = 0;
		double b = 0;
		double input[] = { 0, 0 };
		double output[] = { 0 };

		// for loop for the number of iterations of learning
		for (int i = 0; i < 100; i++) {

			// The net will learn how to simply add two values between 0.5 (input range is
			// 0-0.5)
			// It will output a value between 0 and 1
			a = (int) (Math.random() * 5) / 10.0;
			b = (int) (Math.random() * 5) / 10.0;
			input[0] = a;
			input[1] = b;

			// Target Output (not actually provided to as input when forward propagating)
			output[0] = (a + b);

			// Efficiency testing
			long startTime = System.nanoTime();
			// Send the input array as parameter and forward propagate
			n.forwardPropagate(input);
			// Gradient descent through backPropagate
			n.backPropagate(output);
			long endTime = System.nanoTime();
			
			DecimalFormat df = new DecimalFormat("###.##");
			// Results of feed forward and back propagation
			System.out.println("Time(ns): " + (endTime - startTime) + "     Iteration: " + i + "     Input: " + input[0]
					+ " + " + input[1] + " = " + (int)(output[0]*10.0)/10.0 + "     Prediction: " + n.network.get(2).layer.get(0).activation
					+ "     Error: " + Math.pow(n.network.get(2).layer.get(0).activation - output[0], 2));
		}

	}
}