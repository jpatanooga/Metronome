package tv.floe.metronome.classification.neuralnetworks.math.random;

import java.util.Random;

import tv.floe.metronome.classification.neuralnetworks.core.Connection;
import tv.floe.metronome.classification.neuralnetworks.core.Layer;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;

public class WeightsRandomizer {

   protected Random randomGenerator;

   public WeightsRandomizer() {
       this.randomGenerator = new Random();
   }
   
   public WeightsRandomizer(Random randomGenerator) {
       this.randomGenerator = randomGenerator;
   }    

   public Random getRandomGenerator() {
       return randomGenerator;
   }

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

   protected double nextRandomWeight() {
       return randomGenerator.nextDouble();
   }
}
