package tv.floe.metronome.classification.neuralnetworks.input;

import java.io.Serializable;
import java.util.ArrayList;

import tv.floe.metronome.classification.neuralnetworks.core.Connection;

public abstract class InputFunction implements Serializable {
	
	abstract public double getOutput(ArrayList<Connection> inConnections );

}
