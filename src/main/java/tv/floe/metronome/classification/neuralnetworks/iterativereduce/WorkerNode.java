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
import tv.floe.metronome.classification.neuralnetworks.activation.Tanh;
import tv.floe.metronome.io.records.CachedVector;
import tv.floe.metronome.io.records.CachedVectorReader;
import tv.floe.metronome.io.records.MetronomeRecordFactory;
import tv.floe.metronome.io.records.RecordFactory;
import tv.floe.metronome.io.records.libsvmRecordFactory;

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
public class WorkerNode implements ComputableWorker<NetworkWeightsUpdateable> {

	  private boolean IterationComplete = false;
	  private int CurrentIteration = 0;
	  protected Configuration conf = null;	  
	  protected int NumberIterations = 1;
	  protected String RecordFactoryClassname = "";	  
	  
	  // basic stats tracking
	  Metrics metrics = new Metrics();
	  public double lastRMSE = 0.0;
	boolean hitErrThreshold = false;
	int trainingCompleteEpoch = -1;
	double learningRate = 0;
	int stallMaxEpochs = -1; // default
	double stallMinErrorDelta = -1; // take defaults
	boolean stallBustingOn = true; // defaults to on
	boolean adagradLearningRateOn = false;
	double adagradLearningRateInitSetting = 10;
	
	RecordFactory rec_factory = null; // gotta be dynamically set!
	
	// TODO: fix so its not hardcoded
	TextRecordParser lineParser = new TextRecordParser();
	
	CachedVectorReader cachedVecReader = null; //new CachedVectorReader(lineParser, rec_factory); 
	
	NeuralNetwork nn = null;
	
	private boolean metricsOn = false;
	private String layerNeuronCounts = "2,3,1"; // default XOR network
//	private double learningRate = 0.1d;
	private double trainingErrorThreshold = 0.2d;
	private boolean useVectorCaching = true;
	private String vectorSchema = ""; // tv.floe.metronome.neuralnetwork.conf.InputRecordSchema
	
//	private int inputVectorSize = 0; // in the event you'd use libsvm... FIX?
	
	/**
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
		
		cachedVecReader.Reset();
		
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
			
			
			String marker = "";
			if (hitErrThreshold) {
				marker += ", Hit Err Threshold at: " + this.trainingCompleteEpoch;
			}

			if (bp.checkForLearningStallOut() && false == bp.hasHitMinErrorThreshold()) {
				marker += " [ --- STALL ---]";
				this.nn.randomizeWeights();
				if (this.stallBustingOn) {
					bp.resetStallTracking();
					System.out.println("[ --- STALL WORKER RESET --- ]: " + bp.getSetMaxStalledEpochs());
				}
			}
			
			String alr_debug = bp.DebugAdagrad();
			
			this.metrics.printProgressiveStepDebugMsg(this.CurrentIteration, "Epoch: " + this.CurrentIteration + " > RMSE: " + bp.calcRMSError()  + ", Records Trainined: " + this.cachedVecReader.recordsInCache() + marker + ", ALR: " + alr_debug );
			if (this.metricsOn) {
				bp.getMetrics().PrintMetrics();
			}

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
	      
	      
	    this.NumberIterations = this.conf.getInt("app.iteration.count", 1);
	      
	    this.stallMaxEpochs = this.conf.getInt("tv.floe.metronome.neuralnetwork.conf.stall.maxEpochs", 200);

	    this.stallMinErrorDelta = Double.parseDouble(this.conf.get("tv.floe.metronome.neuralnetwork.conf.stall.minErrorDelta", "0.000001"));
	    
	    String stallBusterOn = this.conf.get("tv.floe.metronome.neuralnetwork.conf.StallBusterOn");
	    if (stallBusterOn != null && stallBusterOn.equals("true")) {
	    	this.stallBustingOn = true;
	    } else {
	    	this.stallBustingOn = false;
	    }
	    
	    
	    
	      this.learningRate = Double.parseDouble(this.conf.get(
		          "tv.floe.metronome.neuralnetwork.conf.LearningRate", "0.1"));

	      // tv.floe.metronome.neuralnetwork.conf.AdagradLearningRateOn
		    String adagradOn = this.conf.get("tv.floe.metronome.neuralnetwork.conf.Adagrad.On");
		    if (adagradOn != null && adagradOn.equals("true")) {
		    	this.adagradLearningRateOn = true;
		    	
		    	this.adagradLearningRateInitSetting = Double.parseDouble(this.conf.get(
				          "tv.floe.metronome.neuralnetwork.conf.Adagrad.LearningRate", "10.0"));
		    	
		    }
	      
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
		bp.setLearningRate(this.learningRate);
		bp.setStallDetectionParams(this.stallMinErrorDelta, this.stallMaxEpochs);
		
		if (this.adagradLearningRateOn) {
			bp.turnOnAdagradLearning(this.adagradLearningRateInitSetting);
			bp.setup(); // we may need to find a better place for this
			System.out.println("Turning on Adagrad Learning...");
		}
		
//		System.out.println("Debug-Stall > stallMinErrordelta: " + this.stallMinErrorDelta);
//		System.out.println("Debug-Stall > stallMaxEpochs: " + this.stallMaxEpochs);
//		System.out.println("Debug-Stall > bp: " + bp.getSetMaxStalledEpochs());
		
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
	    
		NeuralNetworkWeightsDelta global_update = nwu.get();
		
		// TODO: now update the local network
		
		//this.nn = global_update.network;
		this.nn.copyWeightsAndConf( global_update.network );
		
		// this is a hack for now TODO: fix this
		BackPropogationLearningAlgorithm bp = ((BackPropogationLearningAlgorithm)this.nn.getLearningRule());
		bp.setStallDetectionParams(this.stallMinErrorDelta, this.stallMaxEpochs);
		
		//System.out.println("max: " + bp.getSetMaxStalledEpochs());
		
		
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
	    
	    return delta;
	    
	  }	
	
	  protected String LoadStringConfVarOrException(String ConfVarName,
		      String ExcepMsg) throws Exception {
		    
		    if (null == this.conf.get(ConfVarName)) {
		      throw new Exception(ExcepMsg);
		    } else {
		      return this.conf.get(ConfVarName);
		    }
		    
		  }
		  
		  protected int LoadIntConfVarOrException(String ConfVarName, String ExcepMsg)
		      throws Exception {
		    
		    if (null == this.conf.get(ConfVarName)) {
		      throw new Exception(ExcepMsg);
		    } else {
		      return this.conf.getInt(ConfVarName, 0);
		    }
		    
		  }	
	
	  public static void main(String[] args) throws Exception {
		    TextRecordParser parser = new TextRecordParser();
		    WorkerNode wn = new WorkerNode();
		    ApplicationWorker<NetworkWeightsUpdateable> aw = new ApplicationWorker<NetworkWeightsUpdateable>(
		        parser, wn, NetworkWeightsUpdateable.class);
		    
		    ToolRunner.run(aw, args);
	  }
	

}
