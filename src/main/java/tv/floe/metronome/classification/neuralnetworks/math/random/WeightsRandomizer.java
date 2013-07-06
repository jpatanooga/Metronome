package tv.floe.metronome.classification.neuralnetworks.math.random;

import java.util.Random;

import tv.floe.metronome.classification.neuralnetworks.core.Connection;
import tv.floe.metronome.classification.neuralnetworks.core.Layer;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;

public class WeightsRandomizer {

    /* Random number genarator used by randomizers
    */
   protected Random randomGenerator;

   /**
    * Create a new instance of WeightsRandomizer
    */
   public WeightsRandomizer() {
       this.randomGenerator = new Random();
   }
   
   /**
    * Create a new instance of WeightsRandomizer with specified random generator
    * If you use the same random generators, you'll get the same random sequences
    * @param randomGenerator random geneartor to use for randomizing weights
    */    
   public WeightsRandomizer(Random randomGenerator) {
       this.randomGenerator = randomGenerator;
   }    

   /**
    * Gets random generator used to generate random values
    * @return random generator used to generate random values
    */
   public Random getRandomGenerator() {
       return randomGenerator;
   }

   /**
    * Iterate all layers, neurons and connections in network, and randomize connection weights
    * @param neuralNetwork neural network to randomize
    */
   public void randomize(NeuralNetwork neuralNetwork) {
       for (Layer layer : neuralNetwork.getLayers()) {
           for (Neuron neuron : layer.getNeurons()) {
               for (Connection connection : neuron.getInConnections()) {
            	   
            	   if (null == connection.getWeight()) {
            		   System.out.println( "> Conn == null, Layer index: " + layer.getLayerIndex());
            	   }
            	   
                   connection.getWeight().setValue(nextRandomWeight());
               }
           }
       }
   }

   /**
    * Returns next random value from random generator, that will be used to initialize weight
    * @return next random value fro random generator
    */
   protected double nextRandomWeight() {
       return randomGenerator.nextDouble();
   }
}
