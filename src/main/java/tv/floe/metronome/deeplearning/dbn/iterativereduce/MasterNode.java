package tv.floe.metronome.deeplearning.dbn.iterativereduce;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;

import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;

import com.cloudera.iterativereduce.ComputableMaster;
import com.cloudera.iterativereduce.yarn.appmaster.ApplicationMaster;


public class MasterNode implements ComputableMaster<DBNParameterVectorUpdateable> {

	DeepBeliefNetwork dbn_averaged_master = null;
	double trainingErrorThreshold = 0;
	boolean hasHitThreshold = false;
	protected Configuration conf = null;
	
	double learningRate = 0.01;
	int batchSize = 1;
	int numIns = 784;
	int numLabels = 10;
	int[] hiddenLayerSizes = null;
	int n_layers = 1;
	

	@Override
	public void complete(DataOutputStream arg0) throws IOException {
		// TODO Auto-generated method stub
		
		System.out.println( "IR DBN Master Node: Complete!" );
		
	}

	/**
	 * Master::Compute
	 * 
	 * This is where the worker parameter averaged updates come in and are processed
	 * 
	 */
	@Override
	public DBNParameterVectorUpdateable compute(
			Collection<DBNParameterVectorUpdateable> workerUpdates,
			Collection<DBNParameterVectorUpdateable> masterUpdates) {

		DBNParameterVectorUpdateable masterReturnMsg = new DBNParameterVectorUpdateable();
		
		DBNParameterVectorUpdateable firstWorkerMsg = workerUpdates.iterator().next();

		if (null == firstWorkerMsg) {
			
			System.out.println("Can't seem to get the first network weights updateable");
			
		} else {
			
			if (null == this.dbn_averaged_master) {
				
				System.out.println("Building base master MLP network");
				//this.master_nn = new MultiLayerPerceptronNetwork();
				
				// TODO: init dbn_averaged_master
				
		        try {
					//this.master_nn.buildFromConf(first.networkUpdate.network.getConfig());
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
		        
			}
			
		}	
		
		
		
		try {
			//accumNet = NetworkAccumulator.buildAveragingNetworkFromConf(first.networkUpdate.network.getConfig());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
/*		
		if (null == accumNet) {
			System.out.println("Master: Network Accumulator is null! [Error]");
			return null;
		}
*/
		
	    for (DBNParameterVectorUpdateable nn_worker : workerUpdates) {


//	    	accumNet.AccumulateWorkerNetwork(nn_worker.networkUpdate.network);
//	    	avg_rmse += nn_worker.networkUpdate.RMSE;
	    	
	    }
	    
	    // TODO: examine termination conditions
	    
/*	    
	    if (avg_rmse <= this.trainingErrorThreshold && !hasHitThreshold && first.networkUpdate.CurrentIteration > 10) {
	    	
	    	System.out.println("\nMaster hit avg rmse threshold at epoch: " + first.networkUpdate.CurrentIteration + "\n");
	    	
		    for (NetworkWeightsUpdateable nn_worker : workerUpdates) {

		    	System.out.println("worker.rmse: " + nn_worker.networkUpdate.RMSE );
		    }
	    	
	    	
	    	this.hasHitThreshold = true;
	    } else {
	    	
	    	//System.out.println("rmse debug > " + avg_rmse + " <= " + this.trainingErrorThreshold + ", iterations: " + first.networkUpdate.CurrentIteration);
	    	
	    }
*/	    	
	    
	    // TODO: compute average of weights
/*	    try {
			accumNet.AverageNetworkWeights();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
*/	    
	    
		DBNParameterVector dbn_update = new DBNParameterVector();
		
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		this.dbn_averaged_master.write( out );
		dbn_update.dbn_payload = out.toByteArray();
		
		//DBNParameterVectorUpdateable updateable = new DBNParameterVectorUpdateable();
		masterReturnMsg.param_msg = dbn_update;
	    
	    // THIS NEEDS TO BE DONE, probably automated!
	    workerUpdates.clear();
	    masterUpdates.clear();		
		
		
		
		return masterReturnMsg;
	}

	@Override
	public DBNParameterVectorUpdateable getResults() {
		System.out.println("Master >>> getResults() - null!!!");
		return null;
	}

	/**
	 * TODO: finish this up!
	 */
	@Override
	public void setup(Configuration c) {
	
		String useRegularization = "false";
	
	    this.conf = c;
	    
	    try {
	
		      this.learningRate = Double.parseDouble(this.conf.get(
			          "tv.floe.metronome.dbn.conf.LearningRate", "0.001"));
		      
		      this.batchSize = this.conf.getInt("tv.floe.metronome.dbn.conf.batchSize",  1);
		      
		      this.numIns = this.conf.getInt( "tv.floe.metronome.dbn.conf.numberInputs", 784);
		      
		      this.numLabels = this.conf.getInt( "tv.floe.metronome.dbn.conf.numberLabels", 10 );
		      
		      //500, 250, 100
		      String hiddenLayerConfSizes = this.conf.get( "tv.floe.metronome.dbn.conf.hiddenLayerSizes" );
		      
		      String[] layerSizes = hiddenLayerConfSizes.split(",");
		      this.hiddenLayerSizes = new int[ layerSizes.length ];
		      for ( int x = 0; x < layerSizes.length; x++ ) {
		    	  this.hiddenLayerSizes[ x ] = Integer.parseInt( layerSizes[ x ] );
		      }
		      

		
			    useRegularization = this.conf.get("tv.floe.metronome.dbn.conf.useRegularization");
		
			
				
				this.n_layers = hiddenLayerSizes.length;
	      
	
	    } catch (Exception e) {
	      // TODO Auto-generated catch block
	      e.printStackTrace();
	      System.out.println(">> Error loading conf!");
	    }
	    
	    System.out.println( "-----------------------------------------" );
	    System.out.println( "# Master Conf #" );
	    //System.out.println( "Number Iterations: " + this.NumberIterations );
	    System.out.println( "-----------------------------------------\n\n" );
	    		
		
	}

  public static void main(String[] args) throws Exception {
	    MasterNode pmn = new MasterNode();
	    ApplicationMaster< DBNParameterVectorUpdateable > am = new ApplicationMaster< DBNParameterVectorUpdateable >(
	        pmn, DBNParameterVectorUpdateable.class);
	    
	    ToolRunner.run(am, args);
  }

}
