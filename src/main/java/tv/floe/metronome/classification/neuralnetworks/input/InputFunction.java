package tv.floe.metronome.classification.neuralnetworks.input;

import java.util.ArrayList;

import tv.floe.metronome.classification.neuralnetworks.core.Connection;

public abstract class InputFunction {
	abstract public double getOutput(ArrayList<Connection> inConnections );

}
