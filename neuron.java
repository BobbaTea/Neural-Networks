import java.util.ArrayList;

//class for individual neurons
public class neuron {
	
	//activation, net, bias, and weights (between this neuron and next layer neurons)
	public double activation;
	public double net;
	public double bias;
	public ArrayList<Double> weights;

	//constructor to set random values
	neuron(int nextLayer) {
		weights = new ArrayList<Double>();
		activation = 0;
		net = 0;
		bias =  (int)(Math.random()*10.0)/10.0;
		for (int i = 0; i < nextLayer; i++) {
			weights.add((int)(Math.random()*10.0)/10.0);
		}
	}

}
