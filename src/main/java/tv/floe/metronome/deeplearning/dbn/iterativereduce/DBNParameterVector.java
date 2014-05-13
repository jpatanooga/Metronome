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
	
	// worker sends this signal when it has finished N passes of it's split
	// signal from worker->to->master
	public boolean preTrainPhaseComplete = false;
	
	// signal from worker->to->master
	public boolean datasetPassComplete = false;
	
	// master see's that all workers have worker.preTrainPhaseComplete == true
	// master responds with this flag
	// signal: master->to->worker
	public boolean masterSignalToStartFineTunePhase = false;
	
	public boolean masterSignalToStartNextDatasetPass = false;
	
	byte[] dbn_payload = null;
	
	
	public byte[] Serialize() throws IOException {

		// DataOutput d
		
		

		ByteArrayOutputStream out = new ByteArrayOutputStream();
		DataOutput d = new DataOutputStream(out);

		// write the DBN data right into
		// this.dbn.write(out);

		//return this.dbn_payload;
		
		d.writeBoolean( this.datasetPassComplete );
		d.writeBoolean( this.preTrainPhaseComplete );
		
		d.writeBoolean( this.masterSignalToStartFineTunePhase );
		d.writeBoolean( this.masterSignalToStartNextDatasetPass );
		
		d.writeInt( this.dbn_payload.length );
		out.write( this.dbn_payload );
		
		
		return out.toByteArray();
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
		
		

		ByteArrayInputStream b = new ByteArrayInputStream(bytes);
		DataInput in = new DataInputStream(b);

		this.datasetPassComplete = in.readBoolean();
		this.preTrainPhaseComplete = in.readBoolean();
		this.masterSignalToStartFineTunePhase = in.readBoolean();
		this.masterSignalToStartNextDatasetPass = in.readBoolean();
		
		int bytesToRead = in.readInt();
		
		this.dbn_payload = new byte[ bytesToRead ];
		in.readFully( this.dbn_payload, 0, bytesToRead );
		

		// this.dbn.load(b);
		//this.dbn_payload = bytes;

	}

}
