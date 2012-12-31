package tv.floe.metronome.linearregression.iterativereduce;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.classifier.sgd.UniformPrior;

import tv.floe.metronome.io.records.RCV1RecordFactory;
import tv.floe.metronome.io.records.RecordFactory;
import tv.floe.metronome.linearregression.ModelParameters;
import tv.floe.metronome.linearregression.ParallelOnlineLinearRegression;
import tv.floe.metronome.linearregression.ParameterVector;

import com.cloudera.iterativereduce.ComputableMaster;
//import com.cloudera.knittingboar.messages.iterativereduce.ParameterVectorUpdateable;
//import com.cloudera.knittingboar.sgd.iterativereduce.POLRNodeBase;
import com.cloudera.iterativereduce.yarn.appmaster.ApplicationMaster;
/*
import com.cloudera.knittingboar.messages.iterativereduce.ParameterVectorGradient;
import com.cloudera.knittingboar.messages.iterativereduce.ParameterVectorUpdateable;
import com.cloudera.knittingboar.records.CSVBasedDatasetRecordFactory;
import com.cloudera.knittingboar.records.RCV1RecordFactory;
import com.cloudera.knittingboar.records.RecordFactory;
import com.cloudera.knittingboar.records.TwentyNewsgroupsRecordFactory;
import com.cloudera.knittingboar.sgd.GradientBuffer;
import com.cloudera.knittingboar.sgd.POLRModelParameters;
import com.cloudera.knittingboar.sgd.ParallelOnlineLogisticRegression;
import com.cloudera.knittingboar.sgd.iterativereduce.MasterNode;
*/
import com.google.common.collect.Lists;

public class MasterNode extends NodeBase implements
		ComputableMaster<ParameterVectorUpdateable> {


	  private static final Log LOG = LogFactory.getLog(MasterNode.class);
	  
	  ParameterVector global_parameter_vector = null;
	  
//	  private int GlobalMaxPassCount = 0;
	  
	  private int Global_Min_IterationCount = 0;
	  
	  // these are only used for saving the model
	  public ParallelOnlineLinearRegression polr = null;
	  public ModelParameters polr_modelparams;
	  private RecordFactory VectorFactory = null;
	  
	  @Override
	  public ParameterVectorUpdateable compute(
	      Collection<ParameterVectorUpdateable> workerUpdates,
	      Collection<ParameterVectorUpdateable> masterUpdates) {
	    
	    System.out.println("\nMaster Compute: SuperStep - Worker Info ----- ");
	    int x = 0;

	    // reset
	    //this.Global_Min_IterationCount = this.NumberPasses;
	    boolean iterationComplete = true;

	    for (ParameterVectorUpdateable i : workerUpdates) {
	      
	      // not sure we still need this ---------------
/*	      if (i.get().SrcWorkerPassCount > this.GlobalMaxPassCount) {
	        
	        this.GlobalMaxPassCount = i.get().SrcWorkerPassCount;
	        
	      }
	*/      
	      // if any worker is not done with hte iteration, trip the flag
	      if (i.get().IterationComplete == 0 ) {
	        
	        //this.Global_Min_IterationCount = i.get().IterationCount;
	        iterationComplete = false;
	        
	      }      
	      
	      System.out.println("[Master] WorkerReport[" + x + "]: I: " + i.get().CurrentIteration + ", IC: " + i.get().IterationComplete + " Trained Recs: "
	          + i.get().TrainedRecords + " AvgLogLikelihood: "
	          + i.get().AvgLogLikelihood + " PercentCorrect: "
	          + i.get().PercentCorrect);
	   
	      if ( i.get().IterationComplete == 1) {
	        System.out.println( "> worker " + x + " is done with current iteration" );
	      }
	      x++;
	      // accumulate gradient of parameter vectors
	      //this.global_parameter_vector.AccumulateGradient(i.get().parameter_vector);
	      this.global_parameter_vector.AccumulateVector(i.get().parameter_vector.viewRow(0));
	      
	    }
	    
	    // now average the parameter vectors together
	    //this.global_parameter_vector.AverageAccumulations(workerUpdates.size());
	    this.global_parameter_vector.AverageVectors(workerUpdates.size());
	    
	    LOG.debug("Master node accumulating and averaging " + workerUpdates.size()
	        + " worker updates.");
	    
	    
	    
	    
	    ParameterVector vec_msg = new ParameterVector();
//	    vec_msg.GlobalPassCount = this.GlobalMaxPassCount;
	    
	/*    if (iterationComplete) {
	      gradient_msg.IterationComplete = 1;
	      System.out.println( "> Master says: Iteration Complete" );
	    } else {
	      gradient_msg.IterationComplete = 0;
	    }
	    */
	    vec_msg.parameter_vector = this.global_parameter_vector.parameter_vector
	        .clone();
	    
	    ParameterVectorUpdateable return_msg = new ParameterVectorUpdateable();
	    return_msg.set(vec_msg);
	    
	    // set the master copy!
	    this.polr.SetBeta(this.global_parameter_vector.parameter_vector.clone());
	    
	    // THIS NEEDS TO BE DONE, probably automated!
	    workerUpdates.clear();
	    
	    return return_msg;
	  }
	  
	  @Override
	  public ParameterVectorUpdateable getResults() {
	    System.out.println(">>> getResults() - null!!!");
	    return null;
	  }
	  
	  @Override
	  public void setup(Configuration c) {
	    
	    this.conf = c;
	    
	    try {
	      
	      // this is hard set with LR to 2 classes
//	      this.num_categories = this.conf.getInt(
//	          "com.cloudera.knittingboar.setup.numCategories", 2);
	      
	      // feature vector size
	      
	      this.FeatureVectorSize = LoadIntConfVarOrException(
	          "com.cloudera.knittingboar.setup.FeatureVectorSize",
	          "Error loading config: could not load feature vector size");
	      
	      // feature vector size
//	      this.BatchSize = this.conf.getInt(
//	          "com.cloudera.knittingboar.setup.BatchSize", 200);
	      
//	      this.NumberPasses = this.conf.getInt(
//	          "com.cloudera.knittingboar.setup.NumberPasses", 1);
	      this.NumberIterations = this.conf.getInt("app.iteration.count", 1);
	      
	      // protected double Lambda = 1.0e-4;
	      this.Lambda = Double.parseDouble(this.conf.get(
	          "com.cloudera.knittingboar.setup.Lambda", "1.0e-4"));
	      
	      // protected double LearningRate = 50;
	      this.LearningRate = Double.parseDouble(this.conf.get(
	          "com.cloudera.knittingboar.setup.LearningRate", "10"));
	      
	      // local input split path
	      // this.LocalInputSplitPath = LoadStringConfVarOrException(
	      // "com.cloudera.knittingboar.setup.LocalInputSplitPath",
	      // "Error loading config: could not load local input split path");
	      
	      // System.out.println("LoadConfig()");
	      
	      // maps to either CSV, 20newsgroups, or RCV1
	      this.RecordFactoryClassname = LoadStringConfVarOrException(
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
	      
	    } catch (Exception e) {
	      // TODO Auto-generated catch block
	      e.printStackTrace();
	      System.out.println(">> Error loading conf!");
	    }
	    
	    System.out.println( "-----------------------------------------" );
	    System.out.println( "# Master Conf #" );
//	    System.out.println( "Number Iterations: " + this.NumberPasses );
	    System.out.println( "-----------------------------------------\n\n" );
	    
	    this.SetupPOLR();
	    
	  } // setup()
	  
	  public void SetupPOLR() {
	    
/*	    System.err.println("SetupOLR: " + this.num_categories + ", "
	        + this.FeatureVectorSize);
	    LOG.debug("SetupOLR: " + this.num_categories + ", "
	        + this.FeatureVectorSize);
	*/    
//	    this.global_parameter_vector = new GradientBuffer(this.num_categories,
//	        this.FeatureVectorSize);
	
		this.global_parameter_vector = new ParameterVector();  
		  
	    String[] predictor_label_names = this.PredictorLabelNames.split(",");
	    
	    String[] variable_types = this.PredictorVariableTypes.split(",");
	    
	    polr_modelparams = new ModelParameters();
	    polr_modelparams.setTargetVariable(this.TargetVariableName); // getStringArgument(cmdLine,
	                                                                 // target));
	    polr_modelparams.setNumFeatures(this.FeatureVectorSize);
	    polr_modelparams.setUseBias(true); // !getBooleanArgument(cmdLine, noBias));
	    
	    List<String> typeList = Lists.newArrayList();
	    for (int x = 0; x < variable_types.length; x++) {
	      typeList.add(variable_types[x]);
	    }
	    
	    List<String> predictorList = Lists.newArrayList();
	    for (int x = 0; x < predictor_label_names.length; x++) {
	      predictorList.add(predictor_label_names[x]);
	    }
	    
	    polr_modelparams.setTypeMap(predictorList, typeList);
	    polr_modelparams.setLambda(this.Lambda); // based on defaults - match
	                                             // command line
	    polr_modelparams.setLearningRate(this.LearningRate); // based on defaults -
	                                                         // match command line
	    
	    // setup record factory stuff here ---------
	    
	    if (RecordFactory.TWENTYNEWSGROUPS_RECORDFACTORY
	        .equals(this.RecordFactoryClassname)) {
	      
//	      this.VectorFactory = new TwentyNewsgroupsRecordFactory("\t");
	      
	    } else if (RecordFactory.RCV1_RECORDFACTORY
	        .equals(this.RecordFactoryClassname)) {
	      
	      this.VectorFactory = new RCV1RecordFactory();
	      
	    } else {
	      
	      // need to rethink this
/*	      
	      this.VectorFactory = new CSVBasedDatasetRecordFactory(
	          this.TargetVariableName, polr_modelparams.getTypeMap());
	      
	      ((CSVBasedDatasetRecordFactory) this.VectorFactory)
	          .firstLine(this.ColumnHeaderNames);
	*/      
	    }
	    
	    polr_modelparams.setTargetCategories(this.VectorFactory
	        .getTargetCategories());
	    
	    // ----- this normally is generated from the POLRModelParams ------
	    
	    this.polr = new ParallelOnlineLinearRegression(
	        this.FeatureVectorSize, new UniformPrior()).alpha(1).stepOffset(1000)
	        .decayExponent(0.9).lambda(this.Lambda).learningRate(this.LearningRate);
	    
	    polr_modelparams.setPOLR(polr);
	    // this.bSetup = true;
	    
	  }
	  
	  @Override
	  public void complete(DataOutputStream out) throws IOException {
	    // TODO Auto-generated method stub
	    System.out.println("master::complete ");
	    System.out.println("complete-ms:" + System.currentTimeMillis());
	    
	    LOG.debug("Master complete, saving model.");
	    
	    try {
	      this.polr_modelparams.saveTo(out);
	    } catch (Exception ex) {
	      throw new IOException("Unable to save model", ex);
	    }
	  }
	  
	  public static void main(String[] args) throws Exception {
	    MasterNode pmn = new MasterNode();
	    ApplicationMaster<ParameterVectorUpdateable> am = new ApplicationMaster<ParameterVectorUpdateable>(
	        pmn, ParameterVectorUpdateable.class);
	    
	    ToolRunner.run(am, args);
	  }	
	
}
