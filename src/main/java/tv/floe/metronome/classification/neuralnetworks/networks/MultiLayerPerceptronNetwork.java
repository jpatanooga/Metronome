package tv.floe.metronome.classification.neuralnetworks.networks;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import tv.floe.metronome.classification.neuralnetworks.conf.Config;
import tv.floe.metronome.classification.neuralnetworks.core.Layer;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.BiasNeuron;
import tv.floe.metronome.classification.neuralnetworks.learning.BackPropogationLearningAlgorithm;
import tv.floe.metronome.classification.neuralnetworks.math.random.NguyenWidrowRandomizer;

public class MultiLayerPerceptronNetwork extends NeuralNetwork {

	public MultiLayerPerceptronNetwork() {
		
	
	}


	@Override
	public void buildFromConf(Config conf) throws Exception {

		// parse hashtable values
		
		
		
		
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
     this.randomizeWeights();
				
     this.setConfig(conf);
     
	}
	


}
