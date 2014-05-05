package tv.floe.metronome.deeplearning.dbn.iterativereduce;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
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
	

	/**
	 * Q: "is compute() called before complete() is called in last epoch?"
	 * 
	 * 
	 */
	@Override
	public void complete(DataOutputStream osStream) throws IOException {
		
		System.out.println( "IR DBN Master Node: Complete!" );
		this.dbn_averaged_master.write( osStream );
		
		
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

		int[] hiddenLayerSizesTmp = new int[] {1};
		
		ArrayList<DeepBeliefNetwork> workerDBNs = new ArrayList<DeepBeliefNetwork>();
		
	    for (DBNParameterVectorUpdateable dbn_worker : workerUpdates) {

	    	ByteArrayInputStream baInputStream = new ByteArrayInputStream( dbn_worker.param_msg.dbn_payload );
	    	
			DeepBeliefNetwork dbn_worker_deser = new DeepBeliefNetwork(1, hiddenLayerSizesTmp, 1, hiddenLayerSizesTmp.length, null); //1, , 1, hiddenLayerSizes.length, rng );
			dbn_worker_deser.load( baInputStream );
			
			try {
				baInputStream.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			workerDBNs.add(dbn_worker_deser);
	    		    	
	    }
	    
	    // init master w dummy params
	    this.dbn_averaged_master = new DeepBeliefNetwork(1, hiddenLayerSizesTmp, 1, hiddenLayerSizesTmp.length, null);
	    
	    this.dbn_averaged_master.initBasedOn( workerDBNs.get( 0 ) );
	    this.dbn_averaged_master.computeAverageDBNParameterVector(workerDBNs);
	    
	    System.out.println("Master > Parameter Averaged! -------- ");
	    
	    
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
	    System.out.println( "-----------------------------------------\n\n" );
	    		
		
	}

  public static void main(String[] args) throws Exception {
	    MasterNode pmn = new MasterNode();
	    ApplicationMaster< DBNParameterVectorUpdateable > am = new ApplicationMaster< DBNParameterVectorUpdateable >(
	        pmn, DBNParameterVectorUpdateable.class);
	    
	    ToolRunner.run(am, args);
  }

}
