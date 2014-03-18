package tv.floe.metronome.deeplearning.dbn.iterativereduce;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;
import tv.floe.metronome.deeplearning.neuralnetwork.core.LogisticRegression;
import tv.floe.metronome.deeplearning.neuralnetwork.core.NeuralNetworkVectorized;
import tv.floe.metronome.deeplearning.neuralnetwork.layer.HiddenLayer;
import tv.floe.metronome.deeplearning.rbm.RestrictedBoltzmannMachine;

/**
 * Need a local copy of the DBN parameter vector stuff
 * 
 * @author josh
 *
 */
public class DBNParameterVector {

	// public DeepBeliefNetwork dbn = null;
/*
	public HiddenLayer[] hiddenLayers;

	public LogisticRegression logisticRegressionLayer;
	// DA / RBM Layers
	public NeuralNetworkVectorized[] preTrainingLayers;
*/
	
	byte[] dbn_payload = null;
	
	public byte[] Serialize() throws IOException {

		// DataOutput d
/*
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		DataOutput d = new DataOutputStream(out);

		// write the DBN data right into
		// this.dbn.write(out);

		// write in hidden layers
		for (int x = 0; x < this.numberLayers; x++) {

			this.hiddenLayers[x].write(os);

		}

		this.logisticRegressionLayer.write(os);

		// DA / RBM Layers
		for (int x = 0; x < this.numberLayers; x++) {

			((RestrictedBoltzmannMachine) this.preTrainingLayers[x]).write(os);

		}
*/
		return this.dbn_payload;
	}

	/**
	 * We're just saving the bytes ref locally because we'll let the DBN
	 * deserialize later on its own time.
	 * 
	 * @param bytes
	 * @throws IOException
	 */
	public void Deserialize(byte[] bytes) throws IOException {
		// DataInput in) throws IOException {

//		ByteArrayInputStream b = new ByteArrayInputStream(bytes);
//		DataInput in = new DataInputStream(b);

		// this.dbn.load(b);
		this.dbn_payload = bytes;

	}

}
