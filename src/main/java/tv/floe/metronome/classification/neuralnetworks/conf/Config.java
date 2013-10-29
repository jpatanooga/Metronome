package tv.floe.metronome.classification.neuralnetworks.conf;

import java.util.ArrayList;
import java.util.Hashtable;

import org.apache.hadoop.conf.Configuration;

import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;
import tv.floe.metronome.classification.neuralnetworks.input.WeightedSum;
import tv.floe.metronome.classification.neuralnetworks.transfer.Linear;

/**
 * At some point I gotta make this thing configurable
 * 
 * @author josh
 *
 */
public class Config extends Hashtable {
	
/*	
    public String weightsFunction;
    public String summingFunction;
    public String inputFunction;
    public String transferFunction;
    public String neuronType;
    public String useBias;
*/	
	ArrayList<Integer> layerNeuronCounts;

	public Config() {
		
		this.setupDefaults();
		
	}
	
	/**
	 * Call this to prep any values out of the setup config
	 * - also can read values out of a hadoop config
	 * 
	 * @param c
	 */
	public void parse(Configuration c) {

		
		
		
		// parse c here
		
		this.parseLayerNeuronCounts( (String)this.getConfValue("layerNeuronCounts"));
		
	}
	
	private void setupDefaults() {
		
        this.setConfValue("inputFunction", WeightedSum.class);
		this.setConfValue("transferFunction", Linear.class);
		this.setConfValue("neuronType", Neuron.class);
		this.setConfValue("networkType", NeuralNetwork.NetworkType.MULTI_LAYER_PERCEPTRON);
		this.setConfValue("layerNeuronCounts", "2,3,2" );
		
		
	}
	
	private void parseLayerNeuronCounts(String values) {
		
		this.layerNeuronCounts = new ArrayList<Integer>();
		
		String[] parts = values.split(",");
		
		for ( int x = 0; x < parts.length; x++ ) {
			
			int val = Integer.parseInt(parts[x]);
			
			this.layerNeuronCounts.add( val );
			
		//	System.out.println(">> ParseNeuronCounts: (index: " + x + "): " + val );
			
		}
		
	}
	
	public void setLayerNeuronCounts(int[] counts) {
		
		this.layerNeuronCounts = new ArrayList<Integer>();
		for ( int x = 0; x < counts.length; x++ ) { 
			this.layerNeuronCounts.add(counts[x]);
		}
		
	}
	
	public int getLayerNeuronCount(int index) {
		return this.layerNeuronCounts.get(index);
	}
	
	public int getLayerCount() {
		return this.layerNeuronCounts.size();
	}
	
	public void setConfValue(String key, Object value) {

		this.put(key, value);

	}

	public Object getConfValue(String key) {
	      return this.get(key);
	}

	
	
	

}
