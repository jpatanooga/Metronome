package tv.floe.metronome.classification.neuralnetworks.iterativereduce;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;

import tv.floe.metronome.classification.logisticregression.iterativereduce.POLRMasterNode;
import tv.floe.metronome.classification.logisticregression.iterativereduce.ParameterVectorUpdatable;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.io.records.RecordFactory;
import tv.floe.metronome.linearregression.iterativereduce.NodeBase;

import com.cloudera.iterativereduce.ComputableMaster;
import com.cloudera.iterativereduce.yarn.appmaster.ApplicationMaster;

public class MasterNode  extends NodeBase implements ComputableMaster<NetworkWeightsUpdateable> {

	public NeuralNetwork master_nn = null;
	public NeuralNetwork first_worker_copy = null;
	
	@Override
	public void complete(DataOutputStream ds) throws IOException {

	    System.out.println("master::complete (Iterations: " + this.NumberIterations + ")");
	    System.out.println("complete-ms:" + System.currentTimeMillis());
	    
	    
	    
	    //LOG.debug("Master complete, saving model.");
	    
/*	    try {
	      this.polr_modelparams.saveTo(out);
	    } catch (Exception ex) {
	      throw new IOException("Unable to save model", ex);
	    }
*/		
		
	}

	@Override
	public NetworkWeightsUpdateable compute(Collection<NetworkWeightsUpdateable> workerUpdates,
			Collection<NetworkWeightsUpdateable> masterUpdates) {

		NetworkWeightsUpdateable return_msg = new NetworkWeightsUpdateable();
		
		//NeuralNetwork nn_0 = workerUpdates
		
		//System.out.println("Count: " + workerUpdates.size());
		
		
		NetworkWeightsUpdateable first = workerUpdates.iterator().next();
		NetworkAccumulator accumNet = null;
		
		if (null == first) {
			System.out.println("Can't seem to get the first network weights updateable");
		}
		
		if (null == this.first_worker_copy) {
			this.first_worker_copy = first.networkUpdate.network;
		}
		
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
	    	
	    }
	    
	    try {
			accumNet.AverageNetworkWeights();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    
	    //accumNet.
		
	    NeuralNetworkWeightsDelta nnwd = new NeuralNetworkWeightsDelta();
	    nnwd.network = accumNet;
	    
	    return_msg.set(nnwd);
	    
	    this.master_nn = nnwd.network;
		
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
	      
	      // this is hard set with LR to 2 classes
//	      this.num_categories = this.conf.getInt(
	//          "com.cloudera.knittingboar.setup.numCategories", 2);
	      
	      // feature vector size
	      
/*	      this.FeatureVectorSize = LoadIntConfVarOrException(
	          "com.cloudera.knittingboar.setup.FeatureVectorSize",
	          "Error loading config: could not load feature vector size");
	*/      
	      // feature vector size
//	      this.BatchSize = this.conf.getInt(
//	          "com.cloudera.knittingboar.setup.BatchSize", 200);
	      
//	      this.NumberPasses = this.conf.getInt(
//	          "com.cloudera.knittingboar.setup.NumberPasses", 1);
//	      this.NumberIterations = this.conf.getInt("app.iteration.count", 1);
	      
	      // protected double Lambda = 1.0e-4;
//	      this.Lambda = Double.parseDouble(this.conf.get(
//	          "com.cloudera.knittingboar.setup.Lambda", "1.0e-4"));
	      
	      // protected double LearningRate = 50;
//	      this.LearningRate = Double.parseDouble(this.conf.get(
//	          "com.cloudera.knittingboar.setup.LearningRate", "10"));
	      
	      // local input split path
	      // this.LocalInputSplitPath = LoadStringConfVarOrException(
	      // "com.cloudera.knittingboar.setup.LocalInputSplitPath",
	      // "Error loading config: could not load local input split path");
	      
	      // System.out.println("LoadConfig()");
	      
	      // maps to either CSV, 20newsgroups, or RCV1
/*	      this.RecordFactoryClassname = LoadStringConfVarOrException(
	          "com.cloudera.knittingboar.setup.RecordFactoryClassname",
	          "Error loading config: could not load RecordFactory classname");
	      
	      if (this.RecordFactoryClassname.equals(RecordFactory.CSV_RECORDFACTORY)) {
	        
	        // so load the CSV specific stuff ----------
	        System.out
	            .println("----- Loading CSV RecordFactory Specific Stuff -------");
	        // predictor label names
	        this.PredictorLabelNames = LoadStringConfVarOrException(
	            "com.cloudera.knittingboar.setup.PredictorLabelNames",
	            "Error loading config: could not load predictor label names");
	        
	        // predictor var types
	        this.PredictorVariableTypes = LoadStringConfVarOrException(
	            "com.cloudera.knittingboar.setup.PredictorVariableTypes",
	            "Error loading config: could not load predictor variable types");
	        
	        // target variables
	        this.TargetVariableName = LoadStringConfVarOrException(
	            "com.cloudera.knittingboar.setup.TargetVariableName",
	            "Error loading config: Target Variable Name");
	        
	        // column header names
	        this.ColumnHeaderNames = LoadStringConfVarOrException(
	            "com.cloudera.knittingboar.setup.ColumnHeaderNames",
	            "Error loading config: Column Header Names");
	        
	        // System.out.println("LoadConfig(): " + this.ColumnHeaderNames);
	        
	      }
*/	      
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
