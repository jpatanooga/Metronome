package tv.floe.metronome.classification.neuralnetworks.iterativereduce;

import java.io.ByteArrayOutputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Collection;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;

import tv.floe.metronome.classification.logisticregression.iterativereduce.POLRMasterNode;
import tv.floe.metronome.classification.logisticregression.iterativereduce.ParameterVectorUpdatable;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.learning.BackPropogationLearningAlgorithm;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;
import tv.floe.metronome.io.records.RecordFactory;
import tv.floe.metronome.linearregression.iterativereduce.NodeBase;

import com.cloudera.iterativereduce.ComputableMaster;
import com.cloudera.iterativereduce.yarn.appmaster.ApplicationMaster;

public class MasterNode  extends NodeBase implements ComputableMaster<NetworkWeightsUpdateable> {

	public NeuralNetwork master_nn = null;
	double trainingErrorThreshold = 0;
	boolean hasHitThreshold = false;
	
	@Override
	public void complete(DataOutputStream ds) throws IOException {

	    System.out.println("master::complete (Iterations: " + this.NumberIterations + ")");
	    
	    ds.write(this.master_nn.Serialize());
	    
		
	}

	@Override
	public NetworkWeightsUpdateable compute(Collection<NetworkWeightsUpdateable> workerUpdates,
			Collection<NetworkWeightsUpdateable> masterUpdates) {

		NetworkWeightsUpdateable return_msg = new NetworkWeightsUpdateable();
				
		double avg_rmse = 0;
		
		
		NetworkWeightsUpdateable first = workerUpdates.iterator().next();
		NetworkAccumulator accumNet = null;
		
		if (null == first) {
			System.out.println("Can't seem to get the first network weights updateable");
		} else {
			
			if (null == this.master_nn) {
				
				System.out.println("Building base master MLP network");
				this.master_nn = new MultiLayerPerceptronNetwork();
		        try {
					this.master_nn.buildFromConf(first.networkUpdate.network.getConfig());
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
		        
			}
			
		}
/*		
		if (null == this.first_worker_copy) {
			this.first_worker_copy = first.networkUpdate.network;
		}
	*/	
		try {
			accumNet = NetworkAccumulator.buildAveragingNetworkFromConf(first.networkUpdate.network.getConfig());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		if (null == accumNet) {
			System.out.println("Master: Network Accumulator is null! [Error]");
			return null;
		}
		
	    for (NetworkWeightsUpdateable nn_worker : workerUpdates) {

	    	accumNet.AccumulateWorkerNetwork(nn_worker.networkUpdate.network);
	    	avg_rmse += nn_worker.networkUpdate.RMSE;
	    }
	    
	    avg_rmse = avg_rmse / workerUpdates.size();
	    BackPropogationLearningAlgorithm bp = ((BackPropogationLearningAlgorithm)this.master_nn.getLearningRule());
	    bp.getMetrics().setLastRMSE(avg_rmse);
	    
	    if (avg_rmse <= this.trainingErrorThreshold && !hasHitThreshold && first.networkUpdate.CurrentIteration > 10) {
	    	System.out.println("\nMaster hit avg rmse threshold at iteration: " + first.networkUpdate.CurrentIteration + "\n");
	    	this.hasHitThreshold = true;
	    } else {
	    	
	    	//System.out.println("rmse debug > " + avg_rmse + " <= " + this.trainingErrorThreshold + ", iterations: " + first.networkUpdate.CurrentIteration);
	    	
	    }
	    	
	    
	    
	    try {
			accumNet.AverageNetworkWeights();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    
	    //accumNet.
	    
	    this.master_nn.copyWeightsAndConf(accumNet);
		
	    NeuralNetworkWeightsDelta nnwd = new NeuralNetworkWeightsDelta();
	    nnwd.network = this.master_nn;
	    
	    return_msg.set(nnwd);
	    
	    //this.master_nn = nnwd.network;
		
	    // THIS NEEDS TO BE DONE, probably automated!
	    workerUpdates.clear();

		
		return return_msg;
	}

	@Override
	public NetworkWeightsUpdateable getResults() {
		System.out.println(">>> getResults() - null!!!");
		return null;
	}

	@Override
	public void setup(Configuration c) {

		

	    this.conf = c;
	    
	    try {

	    	// this is the target to get the avg rmse under for testing purposes
	    	this.trainingErrorThreshold = Double.parseDouble(this.conf.get(
			          "tv.floe.metronome.neuralnetwork.conf.TrainingErrorThreshold", "0.2"));
	      

	    } catch (Exception e) {
	      // TODO Auto-generated catch block
	      e.printStackTrace();
	      System.out.println(">> Error loading conf!");
	    }
	    
	    System.out.println( "-----------------------------------------" );
	    System.out.println( "# Master Conf #" );
	    System.out.println( "Number Iterations: " + this.NumberIterations );
	    System.out.println( "-----------------------------------------\n\n" );
	    		
		
	}

	  public static void main(String[] args) throws Exception {
		    MasterNode pmn = new MasterNode();
		    ApplicationMaster<NetworkWeightsUpdateable> am = new ApplicationMaster<NetworkWeightsUpdateable>(
		        pmn, NetworkWeightsUpdateable.class);
		    
		    ToolRunner.run(am, args);
	  }



}
