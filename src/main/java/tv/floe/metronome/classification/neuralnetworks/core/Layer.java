package tv.floe.metronome.classification.neuralnetworks.core;

import java.io.Serializable;
import java.util.ArrayList;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;


import tv.floe.metronome.classification.neuralnetworks.conf.Config;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;

public class Layer implements Serializable {

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
		
		//System.out.println("Layer " + layerIndex + ": neurons: " + neuronCount );
		
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
	
	public int getIncomingConnectionCount() {
		
		int connections = 0;
		
		// for each neuron
		for (Neuron neuron : this.layer_neurons) {
			connections += neuron.inConnections.size();
		}
		
		return connections;
	}
	
	/**
	 * This allows us to provide more fine-grain SerDe mechanics
	 * - we get each layer's set of incoming connections as a vector
	 * - each Mahout vector can easily and cleanly be written to the underlying SerDe stream
	 * 
	 * 
	 * Today's Assumptions
	 * - fully connected network
	 * 
	 * @return
	 */
	public Vector getIncomingConnectionsAsVector() {
		
		Vector v_out = new RandomAccessSparseVector( this.getIncomingConnectionCount() );
		int v_index = 0;
		
		for (Neuron neuron : this.layer_neurons) {

			//connections += neuron.inConnections.size();
			for ( int x = 0; x < neuron.inConnections.size(); x++ ) {
				
				v_out.set(v_index, neuron.inConnections.get(x).getWeight().value);
				v_index++;
				
			}
			
		}
		

		return v_out;
		
	}
	
	public void loadConnectingWeights(Vector weights) {
		
		int v_index = 0;
		
		for (Neuron neuron : this.layer_neurons) {

			//connections += neuron.inConnections.size();
			for ( int x = 0; x < neuron.inConnections.size(); x++ ) {
				
				//v_out.set(v_index, neuron.inConnections.get(x).getWeight().value);
				neuron.inConnections.get(x).getWeight().setValue(weights.get(v_index));
				v_index++;
				
			}
			
		}
				
		
	}
	

	
}
