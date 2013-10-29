package tv.floe.metronome.classification.neuralnetworks.iterativereduce;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;

import tv.floe.metronome.classification.neuralnetworks.conf.Config;
import tv.floe.metronome.classification.neuralnetworks.core.Connection;
import tv.floe.metronome.classification.neuralnetworks.core.Layer;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork.NetworkType;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.BiasNeuron;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;
import tv.floe.metronome.classification.neuralnetworks.learning.BackPropogationLearningAlgorithm;
import tv.floe.metronome.classification.neuralnetworks.math.random.NguyenWidrowRandomizer;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;

/**
 * Accumulates and averages multiple networks together
 * 
 * Currently only supports multilayer perceptrons
 * 
 * @author josh
 *
 */
public class NetworkAccumulator extends NeuralNetwork {

	
	private ArrayList<NeuralNetwork> worker_networks = new ArrayList<NeuralNetwork>();

	public void ClearNetworkBuffer() {
		
		this.worker_networks = new ArrayList<NeuralNetwork>();
		
	}
	
	public void clearNetworkConnectionWeights() {
		
		for ( int x = 1; x < this.getLayersCount(); x++ ) {
				        	
            for (Neuron neuron : this.getLayerByIndex(x).getNeurons()) {
            	
                for (Connection connection : neuron.getInConnections()) {
                	
                    connection.getWeight().setValue( 0 );
                    
                }
                
            }
	            
			
		}
		
		
	}

	public void AverageNetworkWeights() throws Exception {

		int denominator = this.worker_networks.size(); 
		
		for ( int x = 1; x < this.getLayersCount(); x++ ) {
			
			this.averageAccumulatedLayer(this.getLayerByIndex(x), denominator);
			
		}
		
		
	}
	
	private void averageAccumulatedLayer(Layer layer, int denominator ) {

		for ( int x = 0; x < layer.getNeuronsCount(); x++ ) {
        	
			//this.accumulateNeuronConnections(worker_layer.getNeuronAt(x), summing_layer.getNeuronAt(x));
			this.averageAccumulatedNeuronConnections(layer.getNeuronAt(x), denominator);
            
        }
		
	}
	
	private void averageAccumulatedNeuronConnections( Neuron accumulatedNeuron, int denominator ) {
		
		for ( int x = 0; x < accumulatedNeuron.inConnections.size(); x++ ) {
        	
            //connection.getWeight().setValue( rnd.nextDouble() );
			accumulatedNeuron.getInConnections().get(x).getWeight().average(denominator);
            
        }
		
		
	}
	
	public void AccumulateWorkerNetwork(NeuralNetwork worker_nn) {
		
		this.worker_networks.add( worker_nn );

		// for each layer starting after the input layer
		for ( int x = 1; x < worker_nn.getLayersCount(); x++ ) {
			
			this.accumulateLayer(worker_nn.getLayerByIndex(x), this.getLayerByIndex(x));
			
		}
		
	}
	
	private void accumulateLayer(Layer worker_layer, Layer summing_layer) {
		
        //for (Neuron neuron : worker_layer.getNeurons()) {
		for ( int x = 0; x < worker_layer.getNeuronsCount(); x++ ) {
        	
			this.accumulateNeuronConnections(worker_layer.getNeuronAt(x), summing_layer.getNeuronAt(x));
            
        }
		
	}
	
	private void accumulateNeuronConnections(Neuron workerNeuron, Neuron summingNeuron) {
		
        //for (Connection connection : worker_layer.getInConnections()) {
		for ( int x = 0; x < workerNeuron.inConnections.size(); x++ ) {
        	
            //connection.getWeight().setValue( rnd.nextDouble() );
			summingNeuron.getInConnections().get(x).getWeight().accumulate( workerNeuron.getInConnections().get(x).getWeight().getValue() );
            
        }
		
	}
	
	
	/**
	 * So this is a poor substitute for a more formal factory setup
	 * - but we're running with it for now
	 * 
	 * @param c
	 * @return
	 * @throws Exception 
	 */
	public static NetworkAccumulator buildAveragingNetworkFromConf(Config c) throws Exception {
		
		if (null != c.getConfValue("networkType")) {
			
			if (c.getConfValue("networkType").equals(NetworkType.MULTI_LAYER_PERCEPTRON)) {
				
				NetworkAccumulator mlp_network = new NetworkAccumulator();
								
				mlp_network.buildFromConf(c);		
				
				return mlp_network;
				
			} else {
				
				// other type of unsupported network -- throw exception and junk
				throw new Exception("Currently unsupported network type");
				
			}
			
		}
		
		throw new Exception("Currently unsupported network type");
		
	}	
	
	@Override
	public void buildFromConf(Config conf) throws Exception {

		this.setNetworkType(NetworkType.MULTI_LAYER_PERCEPTRON);


		Layer layer = Layer.createLayer(conf, 0);
        
        boolean useBiasNeuron = false; // use bias neurons by default
        
        if (null != conf.getConfValue("useBiasNeuron")) {
        	
        	if ( conf.getConfValue("useBiasNeuron").equals("true") ) {
        		useBiasNeuron = true; 
        	}
        	
        }
        

        if (useBiasNeuron) {
        	
//        	System.out.println("Using Bias Neuron ---------- ");
        	layer.addNeuron(new BiasNeuron());
 //       	System.out.println( "> Adding Bias Neuron to Input Layer "  );
        	
        }
		
		
		this.addLayer(layer);

		// create layers
		Layer prevLayer = layer;

		// ################# create the other layers ########################
		
     for (int x = 1; x < conf.getLayerCount(); x++){
              
    	 Integer neuronsNum = conf.getLayerNeuronCount(x);
		
    	 
    	 // createLayer layer
    	 layer = Layer.createLayer(conf, x);

    	 if ( useBiasNeuron && (x < ( conf.getLayerCount() - 1 )) ) {
         
    		 System.out.println( "> Adding Bias Neuron to Layer " + x );
    		 layer.addNeuron(new BiasNeuron());
             
    	 }

    	 this.addLayer(layer);
			
		if (prevLayer != null) {

			NeuralNetwork.ConnectLayers(prevLayer, layer);

		}

		prevLayer = layer;
		
     } // for


     this.completeIOWiring();
     this.setLearningRule(new BackPropogationLearningAlgorithm());
     //this.randomizeWeights( new NguyenWidrowRandomizer( -0.7, 0.7 ) );
				
     this.setConfig(conf);
     
     this.clearNetworkConnectionWeights();
     
	}	

    	
	
	
	
}
