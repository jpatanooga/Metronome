package tv.floe.metronome.classification.neuralnetworks.core;

import java.util.ArrayList;


import tv.floe.metronome.classification.neuralnetworks.conf.Config;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;

public class Layer {

	private NeuralNetwork parentNetwork;

	protected ArrayList<Neuron> layer_neurons;
        
	private int index = 0;
        
    private String label;

    
	public Layer(int layer_index) {
		
		this.index = layer_index;
		this.layer_neurons = new ArrayList<Neuron>();
		
	}

	
	public static Layer createLayer(Config c, int layerIndex) throws Exception {
		
		Layer layer = new Layer(layerIndex);
		
		
		
		int neuronCount = c.getLayerNeuronCount(layerIndex);
		
		for (int x = 0; x < neuronCount; x++ ) {
			
			// create neuron
			Neuron n = Neuron.createNeuron(c, layerIndex);
			// add neuron
			layer.addNeuron(n);
			
		}
		
		return layer;
		
		
	}
	
	public int getLayerIndex() {
		
		return this.index;
		
	}
	
	

	public void setParentNetwork(NeuralNetwork parent) {
		this.parentNetwork = parent;
	}

	public NeuralNetwork getParentNetwork() {
		return this.parentNetwork;
	}

	public final ArrayList<Neuron> getNeurons() {
		return this.layer_neurons;
	}

	public final void addNeuron(Neuron neuron) throws Exception {
            // prevent adding null neurons
            if (neuron == null) { 
                throw new Exception("Neuron cant be null!");
            }
            
            if (neuron.getClass().getSimpleName().equals("InputNeuron") && this.index > 0) {
            	
            	throw new Exception("Neuron cant be InputNeuron on a Layer which is not the input layer (0-index)!");
            	
            }

            if (neuron.getClass().getSimpleName().equals("Neuron") && this.index == 0) {
            	
            	throw new Exception("Neuron cant be Neuron on a Layer which is the input layer (0-index)!");
            	
            }
            
                       
            this.layer_neurons.add(neuron);
            neuron.setParentLayer(this); 
            
	}

	public final void addNeuron(int index, Neuron neuron) throws Exception {

		if ((index >= this.layer_neurons.size()) || (index < 0)) {
            throw new Exception("Specified neuron index position is out of range: "+index);
        } 
    
        // and the neuron is not null
        if (neuron == null) { 
            throw new Exception("Neuron cant be null!");
        }     
        
        //System.out.println("Neuron: " + neuron.getClass().getSimpleName() );
        
        if (neuron.getClass().getSimpleName().equals("InputNeuron") && this.index > 0) {
        	
        	throw new Exception("Neuron cant be InputNeuron on a Layer which is not the input layer (0-index)!");
        	
        }

        if (neuron.getClass().getSimpleName().equals("Neuron") && this.index == 0) {
        	
        	throw new Exception("Neuron cant be Neuron on a Layer which is the input layer (0-index)!");
        	
        }
        
        
        
        this.layer_neurons.add(index, neuron);
        neuron.setParentLayer(this);                
	}

        public void setNeuron(int index, Neuron neuron) throws Exception {
            
            if ((index >= this.layer_neurons.size()) || (index < 0)) {
                throw new Exception("Specified neuron index position is out of range: " + index);
            }

            
            if (neuron == null) {
                throw new Exception("Neuron cant be null!");
            }
            
                                    
            neuron.setParentLayer(this);

            // now safely set new neuron at specified index position
            //this.layer_neurons[index] = neuron;
            this.layer_neurons.set(index, neuron);
        }

	public void removeNeuron(Neuron neuron) {
            int index = indexOf(neuron);
            removeNeuronAt(index);
	}

	public void removeNeuronAt(int index) {

		this.layer_neurons.get(index).removeAllConnections();

		this.layer_neurons.remove(index);
		
	}
        
        public void removeAllNeurons() {
        	
            for (int x = 0; x < layer_neurons.size(); x++) {
            	
                this.removeNeuronAt(x);
                
            }
            
        }

	public Neuron getNeuronAt(int index) {
		
		return this.layer_neurons.get(index);
		
	}

	public int indexOf(Neuron neuron) {
		
		return this.layer_neurons.indexOf(neuron);
        
	}

	public int getNeuronsCount() {
		
		return this.layer_neurons.size();
		
	}


	public void randomizeWeights() {
		for(Neuron neuron : this.layer_neurons) {
			neuron.randomizeWeights();
		}
	}


	public void randomizeWeights(double minWeight, double maxWeight) {
		for(Neuron neuron : this.layer_neurons) {
			neuron.randomizeWeights(minWeight, maxWeight);
		}
	}

	
	public void initializeWeights(double value) {
          for(Neuron neuron : this.layer_neurons) {
              neuron.initWeights(value);
          }
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }
 /*   
    public void ConnectAllNeurons(Layer nextLayer) {
    	
    	
    	
    	
    }
   */ 

	public void calculate() {

		for (Neuron neuron : this.layer_neurons) {
			
                neuron.calcOutput();
                
        }

	}

	public void reset() {
		
		for (Neuron neuron : this.layer_neurons) {
			
			neuron.reset();
			
		}		
	}

	
}
