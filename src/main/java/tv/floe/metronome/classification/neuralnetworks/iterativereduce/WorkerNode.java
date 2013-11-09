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
import tv.floe.metronome.classification.neuralnetworks.learning.BackPropogationLearningAlgorithm;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;
import tv.floe.metronome.classification.neuralnetworks.transfer.Tanh;
import tv.floe.metronome.io.records.CachedVector;
import tv.floe.metronome.io.records.CachedVectorReader;
import tv.floe.metronome.io.records.MetronomeRecordFactory;
import tv.floe.metronome.io.records.RecordFactory;
import tv.floe.metronome.io.records.libsvmRecordFactory;
import tv.floe.metronome.linearregression.iterativereduce.NodeBase;
import tv.floe.metronome.metrics.Metrics;

import com.cloudera.iterativereduce.ComputableWorker;
import com.cloudera.iterativereduce.io.RecordParser;
import com.cloudera.iterativereduce.io.TextRecordParser;
import com.cloudera.iterativereduce.yarn.appworker.ApplicationWorker;

/**
 * TODO
 * - figure out / fix the configuration of the Vectorizer setup
 * - vectorizer needs to accept {K,V} pairs
 * EX: want to be able to read a sequence file of images and train a neural network
 * - would give the great transfer rate of MR, with similar mechanics
 * - more efficient throughput for learning at scale
 * 
 * 
 * 
 * @author josh
 *
 */
public class WorkerNode extends NodeBase implements ComputableWorker<NetworkWeightsUpdateable> {

	  private boolean IterationComplete = false;
	  private int CurrentIteration = 0;
	  
	  // basic stats tracking
	  Metrics metrics = new Metrics();
	  public double lastRMSE = 0.0;
	boolean hitErrThreshold = false;
	int trainingCompleteEpoch = -1;
	
	// do we want to hardcode the record factory type in?
	// TODO: fix this
	//libsvmRecordFactory rec_factory = new libsvmRecordFactory(2);
	RecordFactory rec_factory = null; // gotta be dynamically set!
	
	
	// TODO: fix so its not hardcoded
	TextRecordParser lineParser = new TextRecordParser();
	
	CachedVectorReader cachedVecReader = null; //new CachedVectorReader(lineParser, rec_factory); 
	
	NeuralNetwork nn = null;
	
	private boolean metricsOn = false;
	private String layerNeuronCounts = "2,3,1"; // default XOR network
	private double learningRate = 0.1d;
	private double trainingErrorThreshold = 0.2d;
	private boolean useVectorCaching = true;
	private String vectorSchema = ""; // tv.floe.metronome.neuralnetwork.conf.InputRecordSchema
	
	private int inputVectorSize = 0; // in the event you'd use libsvm... FIX?
	
	/**
	 * Constructor for WorkerNode
	 * - TODO: needs to use conf stuff from YARN/Hadoop
	 * 
	 * - thoughts: gets complicated to config things when there are this many moving parts
	 * 
	 */
	public WorkerNode() {
		
		
	}
	
	
	@Override
	public boolean IncrementIteration() {

		this.CurrentIteration++;

		return false;
	}

	/**
	 * need to fix cached record reader API: .hasNext(), .next(...)
	 * 
	 */
	@Override
	public NetworkWeightsUpdateable compute() {

		// the vector to pull from the local read through cache
		CachedVector cv = new CachedVector( this.nn.getInputsCount(), this.rec_factory.getOutputVectorSize() ) ;// rec_factory.getFeatureVectorSize() );

//		System.out.println("NN Inputs Count: " + this.nn.getInputsCount() );
//		System.out.println("RecFactory Inputs Count: " + this.rec_factory.getInputVectorSize() );
//		System.out.println("RecFactory Outputs Count: " + this.rec_factory.getOutputVectorSize() );
		
		long startMS = System.currentTimeMillis();
		
		cachedVecReader.Reset();
		
		//Vector vec_function_output = new DenseVector( this.nn.getOutputsCount() );
		
		BackPropogationLearningAlgorithm bp = ((BackPropogationLearningAlgorithm)this.nn.getLearningRule());
		bp.clearTotalSquaredError();
		
		
			
			try {
				while (cachedVecReader.next(cv)) {
					
					bp.getMetrics().startTrainingRecordTimer();
					
					this.nn.train(cv.vec_output, cv.vec_input);
					
					bp.getMetrics().stopTrainingRecordTimer();
					
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			// TODO: clean up post-epoch -- this may should be handled via the nn interface?
			bp.completeTrainingEpoch();
			
		//} // if
		
			if (bp.hasHitMinErrorThreshold()) {
				if (!hitErrThreshold) {
					//System.out.println("Hit Min Err Threshold > Epoch: " + this.CurrentIteration );
					trainingCompleteEpoch = this.CurrentIteration;
					hitErrThreshold = true;
				}
				
			}
			
			
			String marker = "";
			if (hitErrThreshold) {
				marker += ", Hit Err Threshold at: " + this.trainingCompleteEpoch;
			}

			if (bp.checkForLearningStallOut()) {
				marker += " [ --- STALL ---]";
				this.nn.randomizeWeights();
				bp.resetStallTracking();
				System.out.println("[ - STALL - ]");
			}
			
			
			this.metrics.printProgressiveStepDebugMsg(this.CurrentIteration, "Epoch: " + this.CurrentIteration + " > RMSE: " + bp.calcRMSError()  + ", Records Trainined: " + this.cachedVecReader.recordsInCache() + marker );
			if (this.metricsOn) {
				bp.getMetrics().PrintMetrics();
			}
		
		
		long totalTime = System.currentTimeMillis() - startMS;
		
		//System.out.println("Worker Iteration Time: " + totalTime + " ms");

		NeuralNetworkWeightsDelta nnwd = new NeuralNetworkWeightsDelta();
		nnwd.network = this.nn;
		nnwd.RMSE = bp.calcRMSError();
		this.lastRMSE = nnwd.RMSE;
		
		NetworkWeightsUpdateable nwu = new NetworkWeightsUpdateable();
		nwu.networkUpdate = nnwd;
		nwu.networkUpdate.CurrentIteration = this.CurrentIteration;
		
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
	 * - we can build out the neural network architecture and vectorizer based on these settings
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
	      
	      this.learningRate = Double.parseDouble(this.conf.get(
		          "tv.floe.metronome.neuralnetwork.conf.LearningRate", "0.1"));

	      this.trainingErrorThreshold = Double.parseDouble(this.conf.get(
	          "tv.floe.metronome.neuralnetwork.conf.TrainingErrorThreshold", "0.2"));
	      
	      //System.out.println("layers: " + this.conf.get("tv.floe.metronome.neuralnetwork.conf.LayerNeuronCounts") );
	      
	    this.layerNeuronCounts = LoadStringConfVarOrException(
		          "tv.floe.metronome.neuralnetwork.conf.LayerNeuronCounts",
		          "Error loading config: could not load Layer Neuron Counts!");
		      
	    String metricsOn = this.conf.get("tv.floe.metronome.neuralnetwork.conf.MetricsOn");
	    if (metricsOn != null && metricsOn.equals("true")) {
	    	this.metricsOn = true;
	    }

	      // maps to either CSV, 20newsgroups, or RCV1
	      this.RecordFactoryClassname = LoadStringConfVarOrException(
	          "tv.floe.metronome.neuralnetwork.conf.RecordFactoryClassname",
	          "Error loading config: could not load RecordFactory classname");
	      
	      
	    		  

	      
	      
//	      if (this.RecordFactoryClassname.equals(RecordFactory.CSV_RECORDFACTORY)) {
	        
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
	        
//	      }
	      
	    } catch (Exception e) {
	      // TODO Auto-generated catch block
	      e.printStackTrace();
	    }
	    		

	    // finish it up!
	    try {
			finishNNSetup();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
	
	public void finishNNSetup() throws Exception {

//	    System.out.println("\n> Conf ------- ");
//	    System.out.println("Layers: " + this.layerNeuronCounts);
//	    System.out.println("> Conf ------- \n");

		
		Config c = new Config();
		c.parse(null); // default layer: 2-3-2
        c.setConfValue("inputFunction", WeightedSum.class);
		c.setConfValue("transferFunction", Tanh.class);
		c.setConfValue("neuronType", Neuron.class);
		c.setConfValue("networkType", NeuralNetwork.NetworkType.MULTI_LAYER_PERCEPTRON);
		c.setConfValue("layerNeuronCounts", this.layerNeuronCounts );
		c.parse(null);
		
		this.nn = new MultiLayerPerceptronNetwork();
		try {
			this.nn.buildFromConf(c);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		// setup the learning rate
		BackPropogationLearningAlgorithm bp = ((BackPropogationLearningAlgorithm)this.nn.getLearningRule());
		bp.setLearningRate(this.LearningRate);
		
		if (this.metricsOn) {
			bp.turnMetricsOn();
		}
		
		
	      if (this.RecordFactoryClassname.equals( "tv.floe.metronome.io.records.MetronomeRecordFactory" )) {

//	    	  System.out.println("Using Metronome Format!");

		      this.vectorSchema = LoadStringConfVarOrException(
		    		  "tv.floe.metronome.neuralnetwork.conf.InputRecordSchema",
		    		  "Error Loading Config: Need a vector schema!" );
	    	  
	    	  
	    	  this.rec_factory = new MetronomeRecordFactory( this.vectorSchema );
	    	  
	    	  
	      } else {
	    	  // default to libsvm format
	    	  
//		      this.inputVectorSize = Integer.parseInt(this.conf.get(
//			          "tv.floe.metronome.neuralnetwork.conf.InputVectorSize", "2"));
	    	  
	    	  
//	    	  System.out.println("Defaulting to Using LibSVM Format!");
	    	  this.rec_factory = new libsvmRecordFactory( this.nn.getInputsCount() );
	      }		
	      
	      this.nn.PrintStats();
		
		
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
		
		NeuralNetworkWeightsDelta global_update = nwu.get();
		
		// TODO: now update the local network
		
		this.nn = global_update.network;
		
		
		
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
//	    delta.PercentCorrect = (new Double(metrics.AvgCorrect * 100))
//	        .floatValue();
//	    delta.TrainedRecords = (new Long(metrics.TotalRecordsProcessed))
//	        .intValue();
	    
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
