package tv.floe.metronome.classification.neuralnetworks.networks;

import java.util.List;
import java.util.Vector;

import tv.floe.metronome.classification.neuralnetworks.conf.Config;
import tv.floe.metronome.classification.neuralnetworks.core.Layer;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.BiasNeuron;
import tv.floe.metronome.classification.neuralnetworks.learning.BackPropogationLearningAlgorithm;
import tv.floe.metronome.classification.neuralnetworks.math.random.NguyenWidrowRandomizer;
import tv.floe.metronome.classification.neuralnetworks.transfer.TransferFunction;
import tv.floe.metronome.classification.neuralnetworks.transfer.TransferFunctionType;

public class MultiLayerPerceptronNetwork extends NeuralNetwork {
	
	
	


	public MultiLayerPerceptronNetwork() {
		
	
	}


	public void buildFromConf(Config conf) throws Exception {

		// set TransferFunction to Linear.class
		
		
		
		// set network type
		this.setNetworkType(NetworkType.MULTI_LAYER_PERCEPTRON);

		// ################# create the input layer ########################
		
                // create input layer
        //        NeuronProperties inputNeuronProperties = new NeuronProperties(InputNeuron.class, Linear.class);
        //        Layer layer = LayerFactory.createLayer(neuronsInLayers.get(0), inputNeuronProperties);


		Layer layer = Layer.createLayer(conf, 0);
        
        boolean useBiasNeuron = false; // use bias neurons by default
        
        if (null != conf.getConfValue("useBiasNeuron")) {
        	
        	if ( conf.getConfValue("useBiasNeuron").equals("true") ) {
        		useBiasNeuron = true; //(Boolean) conf.getConfValue("useBiasNeuron"); //(Boolean)neuronProperties.getProperty("useBias");
        	}
        	
        }
        

        if (useBiasNeuron) {
        	
        	System.out.println("Using Bias Neuron ---------- ");
        	layer.addNeuron(new BiasNeuron());
        	System.out.println( "> Adding Bias Neuron to Input Layer "  );
        	
        }
		
		
		this.addLayer(layer);

		// create layers
		Layer prevLayer = layer;

		// ################# create the other layers ########################
		
     for (int x = 1; x < conf.getLayerCount(); x++){
              
    	 Integer neuronsNum = conf.getLayerNeuronCount(x);
		
    	 
    	 // createLayer layer
    	 layer = Layer.createLayer(conf, x);

         // add one more bias neuron to every layer before the output layer
    	 //if ( useBiasNeuron && (x < ( neuronsInLayers.size() - 1 )) ) {
    	 if ( useBiasNeuron && (x < ( conf.getLayerCount() - 1 )) ) {
         
    		 System.out.println( "> Adding Bias Neuron to Layer " + x );
    		 layer.addNeuron(new BiasNeuron());
             
    	 }

		
    	 // add created layer to network
		
    	 this.addLayer(layer);
			
			
		// createLayer full connectivity between previous and this layer
		if (prevLayer != null) {
			//ConnectionFactory.fullConnect(prevLayer, layer);
			//prevLayer.ConnectAllNeurons(layer);
			NeuralNetwork.ConnectLayers(prevLayer, layer);

		}

		prevLayer = layer;
		
     } // for

		// set input and output cells for network

     // ############### need to break this down and impl our version ###########
     //NeuralNetworkFactory.setDefaultIO(this);
		

     this.completeIOWiring();

     this.setLearningRule(new BackPropogationLearningAlgorithm());
     
     this.randomizeWeights( new NguyenWidrowRandomizer( -0.7, 0.7 ) );
				
     
	}
/*
        public void connectInputsToOutputs() {
            // connect first and last layer
            ConnectionFactory.fullConnect( getLayerAt(0), getLayerAt(getLayersCount()-1) , false);
        }
*/
}
