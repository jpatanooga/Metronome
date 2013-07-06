package tv.floe.metronome.classification.neuralnetworks.input;

import java.util.ArrayList;

import tv.floe.metronome.classification.neuralnetworks.core.Connection;

public class WeightedSum extends InputFunction {

    @Override
    public double getOutput(ArrayList<Connection> inConnections) {
        double output = 0d;

        //System.out.println( "> Weighted Sum ---- " );
        
        for (Connection connection : inConnections) {
            output += connection.getWeightedInput();
            //System.out.println( ">> output partial sum: " + connection.getWeightedInput() );
        }

        return output;
    }

    public static double[] getOutput(double[] inputs, double[] weights) {
        double[] output = new double[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            output[i] += inputs[i] * weights[i];
        }

        return output;
    }
}
