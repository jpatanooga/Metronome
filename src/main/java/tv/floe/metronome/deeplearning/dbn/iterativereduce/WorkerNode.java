package tv.floe.metronome.deeplearning.dbn.iterativereduce;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.List;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;

import tv.floe.metronome.deeplearning.datasets.DataSet;
import tv.floe.metronome.deeplearning.datasets.iterator.impl.MnistHDFSDataSetIterator;
import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;
import tv.floe.metronome.deeplearning.dbn.model.evaluation.ModelTester;
import tv.floe.metronome.io.records.CachedVectorReader;

import com.cloudera.iterativereduce.ComputableWorker;
import com.cloudera.iterativereduce.io.RecordParser;
import com.cloudera.iterativereduce.io.TextRecordParser;
import com.cloudera.iterativereduce.yarn.appworker.ApplicationWorker;

/**
 * Base IterativeReduce worker node
 * 
 * @author josh
 *
 */
public class WorkerNode implements ComputableWorker<DBNParameterVectorUpdateable> {
	
	private static final Log LOG = LogFactory.getLog(WorkerNode.class);
	
	  private enum TrainingState {
		    PRE_TRAIN, FINE_TUNE, TRAINING_COMPLETE
		  };
	
	private TrainingState currentTrainingState = TrainingState.PRE_TRAIN; // always start in PRE_TRAIN mode
	
	protected Configuration conf = null;	  

	DeepBeliefNetwork dbn = null;
	
	private boolean preTrainPhaseComplete = false;
	private boolean fineTunePhaseComplete = false;
	
	TextRecordParser lineParser = new TextRecordParser();
	CachedVectorReader cachedVecReader = null; //new CachedVectorReader(lineParser, rec_factory); 

	private boolean epochComplete = false;
	
	private int completedDatasetEpochs = 0;
	private int currentIteration = 0;
	

	int[] hiddenLayerSizes = { 500, 250, 100 };
	double learningRate = 0.01;
	
	// epochs pertaining to a batch
	int preTrainEpochs = 100;
	int fineTuneEpochs = 100;
	
	// passes over dataset
	int preTrainDatasetPasses = 1;
	//int currentPreTrainDatasetPass = 0;
	
	int fineTuneDatasetPasses = 1;
	
	// not used --- or needs to be calc'd?
	int totalTrainingDatasetSize = 1;	
	
	int batchSize = 1;
	boolean showNetworkStats = true;
	
	int numIns = 784; 
	int numLabels = 10; 

	int n_layers = hiddenLayerSizes.length;
	
	
	RandomGenerator rng = new MersenneTwister(123);
	
	MnistHDFSDataSetIterator hdfs_fetcher = null; //new MnistHDFSDataSetIterator( batchSize, totalNumExamples, txt_reader );
	
	StopWatch watch = new StopWatch();
//	watch.start();

	
	/**
	 * generates the local DBN parameter vector
	 *  
	 * 
	 */
	public DBNParameterVector GenerateParameterVectorUpdate() {

		DBNParameterVector vector = new DBNParameterVector();
		//vector.parameter_vector = this.polr.getBeta().clone(); // this.polr.getGamma().getMatrix().clone();

		ByteArrayOutputStream out = new ByteArrayOutputStream();

		this.dbn.write(out);
		
		
		vector.preTrainPhaseComplete = this.preTrainPhaseComplete;
		vector.dbn_payload = out.toByteArray();
		
		System.out.println( "----- GenerateParameterVectorUpdate -----" );
		


/*
		vector.batchTimeMS = this.lastBatchTimeMS;

		vector.AvgError = (new Double(metrics.AvgError * 100))
				.floatValue();
		vector.TrainedRecords = (new Long(metrics.TotalRecordsProcessed))
				.intValue();
		
*/
		return vector;

	}	
	
	/**
	 * Need to think hard about how we define a "pass through the dataset"
	 * - old definition: "iteration"
	 * - new definition: "epoch"
	 * 
	 */
	@Override
	public boolean IncrementIteration() {
		
		this.currentIteration++;
		//this.currentEpoch++;

		return false;
	}
	
	/**
	 * Run a training pass of a single batch of input records on the DBN
	 * 
	 * TODO
	 * - dileneate between pre-train and finetune pass through data
	 * 		- how?
	 * 
	 * - app.iteration.count
	 * 		- indicates how many times we're going to call the workers
	 * 
	 * - tv.floe.metronome.dbn.conf.batchSize=10
	 * 		- indicates that we're going to only process 10 records in a call to a worker
	 * 
	 * - we could either
	 * 
	 * 		1. make a complete pass through the batches in a split between iterations
	 * 
	 * 			- tends to skew away from good solutions
	 * 
	 * 		2. parameter average between batches
	 *
	 *			-	better quality, but more network overhead
	 *
	 * - if we paramete avg between batches, then our passes over the dataset become
	 * 
	 * 		- total number of examples / batch size
	 * 
	 * - might be pragmatic to let a command line tool calculate iterations
	 * 
	 * 		- given we need to know how many fine tune passes to make as well
	 * 
	 * 
	 * 
	 * 
	 * 
	 * 
	 * 
	 * 
	 */
	@Override
	public DBNParameterVectorUpdateable compute() {
		
		// TODO: setup a cached vector system from hdfs for batches
						
	//	System.out.println("Worker > Compute()");

		
		int recordsProcessed = 0;
		
		StopWatch batchWatch = new StopWatch();
		
		DataSet hdfs_recordBatch = null; //this.hdfs_fetcher.next();
		
		
//		if (hdfs_recordBatch.getFirst().numRows() > 0) {
//		do  {
		
		if ( TrainingState.PRE_TRAIN == this.currentTrainingState ) {
		
 			if ( this.hdfs_fetcher.hasNext() ) {
				
				hdfs_recordBatch = this.hdfs_fetcher.next();
				
				
				
				if (hdfs_recordBatch.getFirst().numRows() > 0) {
					
					if (hdfs_recordBatch.getFirst().numRows() < this.batchSize) {
						
						System.out.println( "Worker > PreTrain: [Jagged End of Split: Skipped] Processed Total " + recordsProcessed + " Total Time " + watch.toString() );
						
						
					} else {
						
						// calc stats on number records processed
						recordsProcessed += hdfs_recordBatch.getFirst().numRows();
						
						//System.out.println( "PreTrain: Batch Size: " + hdfs_recordBatch.getFirst().numRows() );
						
						batchWatch.reset();
						
						batchWatch.start();
				
						this.dbn.preTrain( hdfs_recordBatch.getFirst(), 1, this.learningRate, this.preTrainEpochs);
						
						batchWatch.stop();
		
						System.out.println( "Worker > PreTrain: Batch Mode, Processed Total " + recordsProcessed + ", Batch Time " + batchWatch.toString() + " Total Time " + watch.toString() );
		
					} // if
					
					
				} else {
				
					// in case we get a blank line
					System.out.println( "Worker > PreTrain > Idle pass, no records left to process in phase" );
					
				}
				
			}
			
		//	System.out.println( "Worker > Check PreTrain completion > completedEpochs: " + this.completedDatasetEpochs + ", preTrainDatasetPasses: " + this.preTrainDatasetPasses );
			
			// check for completion of split, to signal master on state change
			if (false == this.hdfs_fetcher.hasNext() && this.completedDatasetEpochs + 1 >= this.preTrainDatasetPasses ) {
				
				this.preTrainPhaseComplete = true;
			//	System.out.println( "Worker > Completion of pre-train phase" );
				
			}
			
					
		
		} else if ( TrainingState.FINE_TUNE == this.currentTrainingState) {
			
			//System.out.println( "DBN Network Stats:\n" + dbn.generateNetworkSizeReport() );

			if ( this.hdfs_fetcher.hasNext() ) {
				
				hdfs_recordBatch = this.hdfs_fetcher.next();
				
				if (hdfs_recordBatch.getFirst().numRows() > 0) {
					
					if (hdfs_recordBatch.getFirst().numRows() < this.batchSize) {
						
						System.out.println( "Worker > FineTune: [Jagged End of Split: Skipped] Processed Total " + recordsProcessed + " Total Time " + watch.toString() );

					} else {
						
						batchWatch.reset();
						
						batchWatch.start();
						
						dbn.finetune( hdfs_recordBatch.getSecond(), learningRate, fineTuneEpochs );
						
						batchWatch.stop();
						
						System.out.println( "Worker > FineTune > Batch Mode, Processed Total " + recordsProcessed + ", Batch Time " + batchWatch.toString() + " Total Time " + watch.toString() );
						
					}
					
				} else {
					
				//	System.out.println( "Worker > FineTune > Idle pass, no records left to process in phase" );
					
					
				}
				
			} else {
				
			//	System.out.println( "Worker > FineTune > Alt > [Split Complete, IDLE] > Total Time " + watch.toString() );
				
			}
				
		} else {
			
			// System.err.println( "We're in some impossible training state for this worker" );
		//	System.out.println( "Worker > FineTune > Complete > [Split Complete, IDLE] > Total Time " + watch.toString() );
			
		}

/*		
		try {
			System.out.println(" ----------- Worker model Eval ---------- ");
			ModelTester.evaluateModel( hdfs_recordBatch.getFirst(), hdfs_recordBatch.getSecond(), dbn);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
*/		
		

		// this is a clunky way to do this. dont judge me, working fast here.
		DBNParameterVector dbn_update = new DBNParameterVector();
		dbn_update.preTrainPhaseComplete = this.preTrainPhaseComplete;
		
		if (false == this.hdfs_fetcher.hasNext()) {

		//	System.out.println( "Worker > Dataset Pass Complete" );
			dbn_update.datasetPassComplete = true;
			
		} else {
			
		//	System.out.println( "Worker > Dataset Pass NOT Complete" );
			dbn_update.datasetPassComplete = false;
			
		}
		
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		this.dbn.write(out);
		dbn_update.dbn_payload = out.toByteArray();
		
		DBNParameterVectorUpdateable updateable = new DBNParameterVectorUpdateable();
		updateable.param_msg = dbn_update;
		
		return updateable;
	}
	
	@Override
	public DBNParameterVectorUpdateable compute(List<DBNParameterVectorUpdateable> arg0) {
		
		return compute();
		
	}
	
	@Override
	public DBNParameterVectorUpdateable getResults() {
		
		return new DBNParameterVectorUpdateable( this.GenerateParameterVectorUpdate() );
	
	}
	
	/**
	 * TODO: re-wire this to read blocks of records into a Matrix
	 * 
	 */
	@Override
	public void setRecordParser(RecordParser lineParser) {

		try {
			// Q: is totalTrainingDatasetSize actually used anymore?
			this.hdfs_fetcher = new MnistHDFSDataSetIterator( this.batchSize, this.totalTrainingDatasetSize, (TextRecordParser)lineParser );
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	/**
	 * Setup the local DBN instance based on conf params
	 * 
	 */
	@Override
	public void setup(Configuration c) {

	    this.conf = c;
	    
	    String useRegularization = "false";
	    
	    try {
	      
	      this.learningRate = Double.parseDouble(this.conf.get(
		          "tv.floe.metronome.dbn.conf.LearningRate", "0.01"));
	      
	      this.batchSize = this.conf.getInt( "tv.floe.metronome.dbn.conf.batchSize",  1);
	      
	      this.preTrainDatasetPasses = this.conf.getInt( "tv.floe.metronome.dbn.conf.pretrain.passes", 1 );
	      
	      this.fineTuneDatasetPasses = this.conf.getInt( "tv.floe.metronome.dbn.conf.finetune.passes", 1 );
	      
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
			
			this.dbn = new DeepBeliefNetwork( numIns, hiddenLayerSizes, numLabels, n_layers, rng ); //, Matrix input, Matrix labels);
	
			// default it to off
			this.dbn.useRegularization = false;
			
			if (useRegularization != null && useRegularization.equals("true")) {
		    	this.dbn.useRegularization = true;
		    //	System.out.println(">>> Turning regularization ON!");
		    }
			
			this.dbn.setSparsity( Double.parseDouble( this.conf.get( "tv.floe.metronome.dbn.conf.sparsity", "0.01") ) );
			this.dbn.setMomentum( Double.parseDouble( this.conf.get( "tv.floe.metronome.dbn.conf.momentum", "0" ) ) );		
			
	      
	      
	    } catch (Exception e) {
	      // TODO Auto-generated catch block
	      e.printStackTrace();
	    }
	   		

	    this.watch.start();
	
		
	}
	
	/**
	 * Collect the update from the master node and apply it to the local 
	 * parameter vector
	 * 
	 * TODO: check the state changes of the incoming message!
	 * 
	 */
	@Override
	public void update(DBNParameterVectorUpdateable master_update_updateable) {

		DBNParameterVector master_update = master_update_updateable.get();
		
		ByteArrayInputStream b = new ByteArrayInputStream( master_update.dbn_payload );
		
		// now update the local DBN worker instance
		this.dbn.load(b);
		
		// TODO: check the message for a state change
		
		if ( true == master_update.masterSignalToStartNextDatasetPass ) {
			
			this.completedDatasetEpochs++;
			this.hdfs_fetcher.reset();
			
		//	System.out.println( "Worker > update > starting new data set pass: " + this.completedDatasetEpochs );
			
			if ( this.completedDatasetEpochs >= (this.fineTuneDatasetPasses + this.preTrainDatasetPasses) ) {
				
				// we are done!
				this.currentTrainingState = TrainingState.TRAINING_COMPLETE;
			//	System.out.println( "Worker > Completely done" );
				
			} else if ( this.completedDatasetEpochs >= this.preTrainDatasetPasses && true == master_update.masterSignalToStartFineTunePhase && this.currentTrainingState == TrainingState.PRE_TRAIN ) {

				this.preTrainPhaseComplete = true;
				this.fineTunePhaseComplete = false;

				this.currentTrainingState = TrainingState.FINE_TUNE;
			//	System.out.println( "\n\nWorker > Signaled to move into fine tune phase\n" );
				
			}
			
			
		} else {
			
		//	System.out.println( "Worker > update > not yet time to start next dataset pass" );
			
		}
		/*
		if (true == master_update.masterSignalToStartFineTunePhase && TrainingState.PRE_TRAIN == this.currentTrainingState) {
			
			this.preTrainPhaseComplete = true;
			this.fineTunePhaseComplete = false;
			this.currentTrainingState = TrainingState.FINE_TUNE;
			
			System.out.println( "Worker > Moving into the FineTune phase based on master signal" );
			
			if (false == this.hdfs_fetcher.hasNext()) {
				System.out.println( "\n\n\nWorker > Resetting HDFS Record Reader" );
				this.hdfs_fetcher.reset();
			} else {
				
				System.err.println("Worker > ERR > had more records to process in a state change? How?");
				
			}
			
			
		}
		*/
		
		
	}
	
	

	protected String LoadStringConfVarOrException(String ConfVarName, String ExcepMsg) throws Exception {
		    
		if (null == this.conf.get(ConfVarName)) {
			
			throw new Exception(ExcepMsg);
			
		} else {
			
			return this.conf.get(ConfVarName);
			
		}
		    
	}
  
	protected int LoadIntConfVarOrException(String ConfVarName, String ExcepMsg) throws Exception {
	    
		if (null == this.conf.get(ConfVarName)) {

			throw new Exception(ExcepMsg);
			
		} else {
			
			return this.conf.getInt(ConfVarName, 0);
			
		}
	    
	}	

	public static void main(String[] args) throws Exception {

		TextRecordParser parser = new TextRecordParser();
		WorkerNode wn = new WorkerNode();
		ApplicationWorker<DBNParameterVectorUpdateable> aw = new ApplicationWorker<DBNParameterVectorUpdateable>(parser, wn, DBNParameterVectorUpdateable.class);
			    
		ToolRunner.run(aw, args);
		
	}
	


}
