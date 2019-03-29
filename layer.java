package feedforward;

import java.util.ArrayList;

//class for layers of neurons
public class layer {
	ArrayList<neuron> layer;
	
	public layer(int numNeurons, int nextLayer) {
	    layer = new ArrayList<neuron>();
	    for (int i = 0; i < numNeurons; i++) {
	        layer.add(new neuron(nextLayer));
	    }
	}
}