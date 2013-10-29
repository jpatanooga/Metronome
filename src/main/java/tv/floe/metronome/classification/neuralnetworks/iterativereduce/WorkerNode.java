package tv.floe.metronome.classification.neuralnetworks.iterativereduce;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import tv.floe.metronome.classification.logisticregression.metrics.POLRMetrics;
import tv.floe.metronome.classification.neuralnetworks.conf.Config;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;
import tv.floe.metronome.classification.neuralnetworks.input.WeightedSum;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;
import tv.floe.metronome.classification.neuralnetworks.transfer.Tanh;
import tv.floe.metronome.io.records.CachedVector;
import tv.floe.metronome.io.records.CachedVectorReader;
import tv.floe.metronome.io.records.RecordFactory;
import tv.floe.metronome.io.records.libsvmRecordFactory;
import tv.floe.metronome.linearregression.iterativereduce.NodeBase;

import com.cloudera.iterativereduce.ComputableWorker;
import com.cloudera.iterativereduce.io.RecordParser;
import com.cloudera.iterativereduce.io.TextRecordParser;
import com.cloudera.iterativereduce.yarn.appworker.ApplicationWorker;

public class WorkerNode extends NodeBase implements ComputableWorker<NetworkWeightsUpdateable> {

	  private boolean IterationComplete = false;
	  private int CurrentIteration = 0;
	  
	  // basic stats tracking
	  POLRMetrics metrics = new POLRMetrics();
	
	
	// do we want to hardcode the record factory type in?
	// TODO: fix this
	libsvmRecordFactory rec_factory = new libsvmRecordFactory(2);
	// TODO: fix so its not hardcoded
	TextRecordParser lineParser = new TextRecordParser();
	
	CachedVectorReader cachedVecReader = null; //new CachedVectorReader(lineParser, rec_factory); 
	
	NeuralNetwork nn = null;
	
	/**
	 * Constructor for WorkerNode
	 * - TODO: needs to use conf stuff from YARN/Hadoop
	 * 
	 */
	public WorkerNode() {
		
		Config c = new Config();
		c.parse(null); // default layer: 2-3-2
        c.setConfValue("inputFunction", WeightedSum.class);
		c.setConfValue("transferFunction", Tanh.class);
		c.setConfValue("neuronType", Neuron.class);
		c.setConfValue("networkType", NeuralNetwork.NetworkType.MULTI_LAYER_PERCEPTRON);
		c.setConfValue("layerNeuronCounts", "2,3,1" );
		c.parse(null);
		
		this.nn = new MultiLayerPerceptronNetwork();
		try {
			this.nn.buildFromConf(c);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	
	@Override
	public boolean IncrementIteration() {
		// TODO Auto-generated method stub
		return false;
	}

	/**
	 * need to fix cached record reader API: .hasNext(), .next(...)
	 * 
	 */
	@Override
	public NetworkWeightsUpdateable compute() {

		// the vector to pull from the local read through cache
		CachedVector cv = new CachedVector( 2 ) ;// rec_factory.getFeatureVectorSize() );

		long startMS = System.currentTimeMillis();
		
		cachedVecReader.Reset();
		
		Vector vec_function_output = new DenseVector(1);
		//vec_function_output.set(0, 1);
		
		//System.out.println("Input Neurons: " + this.nn.getLayerByIndex(0).getNeuronsCount());
		
		try {
			while (cachedVecReader.next(cv)) {
				
				//System.out.println( "Worker > CachedVector > Label: " + cv.label + ", Features: " + cv.vec.toString() ) ;
				vec_function_output.set(0, cv.label);
				//mlp_network.train(v0_out, v0);
				this.nn.train(vec_function_output, cv.vec);
				
//			record_count++;
				
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		long totalTime = System.currentTimeMillis() - startMS;
		
		//System.out.println("Worker Iteration Time: " + totalTime + " ms");

		NeuralNetworkWeightsDelta nnwd = new NeuralNetworkWeightsDelta();
		nnwd.network = this.nn;
		
		NetworkWeightsUpdateable nwu = new NetworkWeightsUpdateable();
		nwu.networkUpdate = nnwd;
		
		return nwu;
	}

	/**
	 * Legacy api support
	 */
	@Override
	public NetworkWeightsUpdateable compute(List<NetworkWeightsUpdateable> arg0) {
		return compute();
	}

	@Override
	public NetworkWeightsUpdateable getResults() {
	    return new NetworkWeightsUpdateable(GenerateUpdate());
	}

	@Override
	public void setRecordParser(RecordParser rp) {
		this.lineParser = (TextRecordParser) rp;
		//System.out.println("setting record parser: " + rp.getClass().toString());
		this.cachedVecReader = new CachedVectorReader(lineParser, rec_factory);
	}

	/**
	 * Setup the record factory and record reader
	 * 
	 */
	@Override
	public void setup(Configuration c) {
		

	    this.conf = c;
	    
	    try {
	      
//	      this.num_categories = this.conf.getInt(
//	          "com.cloudera.knittingboar.setup.numCategories", 2);
	      
	      // feature vector size
	      
//	      this.FeatureVectorSize = LoadIntConfVarOrException(
//	          "com.cloudera.knittingboar.setup.FeatureVectorSize",
//	          "Error loading config: could not load feature vector size");
	      
	      // feature vector size
//	      this.BatchSize = this.conf.getInt(
//	          "com.cloudera.knittingboar.setup.BatchSize", 200);
	      
//	      this.NumberPasses = this.conf.getInt(
//	          "com.cloudera.knittingboar.setup.NumberPasses", 1);
	      // app.iteration.count
	    this.NumberIterations = this.conf.getInt("app.iteration.count", 1);
	      
	      // protected double Lambda = 1.0e-4;
//	      this.Lambda = Double.parseDouble(this.conf.get(
//	          "com.cloudera.knittingboar.setup.Lambda", "1.0e-4"));
	      
	      // protected double LearningRate = 50;
//	      this.LearningRate = Double.parseDouble(this.conf.get(
//	          "com.cloudera.knittingboar.setup.LearningRate", "10"));
	      
	      // maps to either CSV, 20newsgroups, or RCV1
	      this.RecordFactoryClassname = LoadStringConfVarOrException(
	          "com.cloudera.knittingboar.setup.RecordFactoryClassname",
	          "Error loading config: could not load RecordFactory classname");
	      
	      if (this.RecordFactoryClassname.equals(RecordFactory.CSV_RECORDFACTORY)) {
	        
	        // so load the CSV specific stuff ----------
	        
	        // predictor label names
/*	        this.PredictorLabelNames = LoadStringConfVarOrException(
	            "com.cloudera.knittingboar.setup.PredictorLabelNames",
	            "Error loading config: could not load predictor label names");
	        
	        // predictor var types
	        this.PredictorVariableTypes = LoadStringConfVarOrException(
	            "com.cloudera.knittingboar.setup.PredictorVariableTypes",
	            "Error loading config: could not load predictor variable types");
	*/        
	        // target variables
/*	        this.TargetVariableName = LoadStringConfVarOrException(
	            "com.cloudera.knittingboar.setup.TargetVariableName",
	            "Error loading config: Target Variable Name");
	        
	        // column header names
	        this.ColumnHeaderNames = LoadStringConfVarOrException(
	            "com.cloudera.knittingboar.setup.ColumnHeaderNames",
	            "Error loading config: Column Header Names");
	*/        
	        // System.out.println("LoadConfig(): " + this.ColumnHeaderNames);
	        
	      }
	      
	    } catch (Exception e) {
	      // TODO Auto-generated catch block
	      e.printStackTrace();
	    }
	    		
		
		
		
	}

	/**
	 * TODO: finish this
	 * 
	 */
	@Override
	public void update(NetworkWeightsUpdateable nwu) {
	    
/*		ParameterVector global_update = t.get();
	    
	    // set the local parameter vector to the global aggregate ("beta")
	    this.polr.SetBeta(global_update.parameter_vector);
	    
	    // update global count
	    this.GlobalBatchCountForIteration = global_update.GlobalPassCount;
	*/
		
//		NeuralNetworkWeightsDelta global_update = nwu.get();
		
		// TODO: now update the local network
		
		
		
		
	}
	
	/**
	 * Do we need to clone the nn object?
	 * 
	 * @return
	 */
	public NeuralNetworkWeightsDelta GenerateUpdate() {
	    
		NeuralNetworkWeightsDelta delta = new NeuralNetworkWeightsDelta();
		delta.network = this.nn; //this.polr.getBeta().clone(); // this.polr.getGamma().getMatrix().clone();
		//delta.SrcWorkerPassCount = this.LocalBatchCountForIteration;
	    
	    if (this.lineParser.hasMoreRecords()) {
	    	delta.IterationComplete = 0;
	    } else {
	    	delta.IterationComplete = 1;
	    }
	    
	    delta.CurrentIteration = this.CurrentIteration;
	    
//	    delta.AvgLogLikelihood = (new Double(metrics.AvgLogLikelihood))
//	        .floatValue();
	    delta.PercentCorrect = (new Double(metrics.AvgCorrect * 100))
	        .floatValue();
	    delta.TrainedRecords = (new Long(metrics.TotalRecordsProcessed))
	        .intValue();
	    
	    return delta;
	    
	  }	
	
	  public static void main(String[] args) throws Exception {
		    TextRecordParser parser = new TextRecordParser();
		    WorkerNode wn = new WorkerNode();
		    ApplicationWorker<NetworkWeightsUpdateable> aw = new ApplicationWorker<NetworkWeightsUpdateable>(
		        parser, wn, NetworkWeightsUpdateable.class);
		    
		    ToolRunner.run(aw, args);
	  }
	

}
