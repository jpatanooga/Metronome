/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package tv.floe.metronome.classification.logisticregression.iterativereduce;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.UniformPrior;
import org.apache.mahout.math.DenseMatrix;

import tv.floe.metronome.classification.logisticregression.POLRModelParameters;
import tv.floe.metronome.classification.logisticregression.ParallelOnlineLogisticRegression;
import tv.floe.metronome.io.records.RCV1RecordFactory;
import tv.floe.metronome.io.records.RecordFactory;

//import com.cloudera.knittingboar.yarn.appmaster.ComputableMaster;
import com.cloudera.iterativereduce.yarn.appmaster.ApplicationMaster;
import com.cloudera.iterativereduce.ComputableMaster;

//import com.cloudera.iterativereduce.yarn

import com.google.common.collect.Lists;

/**
 * Master node for the IR-KnittingBoar YARN process - coordinates the parallel
 * SGD process amongst many workers - gets the parameter vector updates from
 * many workers and averages them together, sending them back to the workers
 * 
 * 
 * @author jpatterson
 * 
 */
public class POLRMasterNode extends POLRNodeBase implements
    ComputableMaster<ParameterVectorUpdatable> {
  
  private static final Log LOG = LogFactory.getLog(POLRMasterNode.class);
  
  ParameterVector global_parameter_vector = null;
  
  private int GlobalMaxPassCount = 0;
  
  private int Global_Min_IterationCount = 0;
  
  // these are only used for saving the model
  public ParallelOnlineLogisticRegression polr = null;
  public POLRModelParameters polr_modelparams;
  private RecordFactory VectorFactory = null;
  
  @Override
  public ParameterVectorUpdatable compute(
      Collection<ParameterVectorUpdatable> workerUpdates,
      Collection<ParameterVectorUpdatable> masterUpdates) {
    
    System.out.println("\nMaster Compute: SuperStep - Worker Info ----- ");
    int x = 0;

    // reset
    //this.Global_Min_IterationCount = this.NumberPasses;
    boolean iterationComplete = true;
    this.global_parameter_vector.parameter_vector = new DenseMatrix(this.num_categories - 1, this.FeatureVectorSize);

    for (ParameterVectorUpdatable i : workerUpdates) {
      
      // not sure we still need this ---------------
      if (i.get().SrcWorkerPassCount > this.GlobalMaxPassCount) {
        
        this.GlobalMaxPassCount = i.get().SrcWorkerPassCount;
        
      }
      
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
      this.global_parameter_vector.AccumulateParameterVector(i.get().parameter_vector);
      
    }
    
    // now average the parameter vectors together
    //this.global_parameter_vector.AverageAccumulations(workerUpdates.size());
    this.global_parameter_vector.AverageParameterVectors(workerUpdates.size());
    
    LOG.debug("Master node accumulating and averaging " + workerUpdates.size()
        + " worker updates.");
    
    
    
    
    ParameterVector gradient_msg = new ParameterVector();
    gradient_msg.GlobalPassCount = this.GlobalMaxPassCount;
    
/*    if (iterationComplete) {
      gradient_msg.IterationComplete = 1;
      System.out.println( "> Master says: Iteration Complete" );
    } else {
      gradient_msg.IterationComplete = 0;
    }
    */
    gradient_msg.parameter_vector = this.global_parameter_vector.parameter_vector.clone();
    
    
    
    ParameterVectorUpdatable return_msg = new ParameterVectorUpdatable();
    return_msg.set(gradient_msg);
    
    // set the master copy!
    this.polr.SetBeta(this.global_parameter_vector.parameter_vector.clone());
    
    // THIS NEEDS TO BE DONE, probably automated!
    workerUpdates.clear();
    
    return return_msg;
  }
  
  @Override
  public ParameterVectorUpdatable getResults() {
    System.out.println(">>> getResults() - null!!!");
    return null;
  }
  
  @Override
  public void setup(Configuration c) {
    
    this.conf = c;
    
    try {
      
      // this is hard set with LR to 2 classes
      this.num_categories = this.conf.getInt(
          "com.cloudera.knittingboar.setup.numCategories", 2);
      
      // feature vector size
      
      this.FeatureVectorSize = LoadIntConfVarOrException(
          "com.cloudera.knittingboar.setup.FeatureVectorSize",
          "Error loading config: could not load feature vector size");
      
      // feature vector size
//      this.BatchSize = this.conf.getInt(
//          "com.cloudera.knittingboar.setup.BatchSize", 200);
      
//      this.NumberPasses = this.conf.getInt(
//          "com.cloudera.knittingboar.setup.NumberPasses", 1);
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
    System.out.println( "Number Iterations: " + this.NumberIterations );
    System.out.println( "-----------------------------------------\n\n" );
    
    this.SetupPOLR();
    
  } // setup()
  
  public void SetupPOLR() {
    
    System.err.println("SetupOLR: " + this.num_categories + ", "
        + this.FeatureVectorSize);
    LOG.debug("SetupOLR: " + this.num_categories + ", "
        + this.FeatureVectorSize);
    
    this.global_parameter_vector = new ParameterVector(); //this.num_categories,
        //this.FeatureVectorSize);
    
    String[] predictor_label_names = this.PredictorLabelNames.split(",");
    
    String[] variable_types = this.PredictorVariableTypes.split(",");
    
    polr_modelparams = new POLRModelParameters();
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
/*    
    if (RecordFactory.TWENTYNEWSGROUPS_RECORDFACTORY
        .equals(this.RecordFactoryClassname)) {
      
      this.VectorFactory = new TwentyNewsgroupsRecordFactory("\t");
      
    } else */
    if (RecordFactory.RCV1_RECORDFACTORY
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
    
    this.polr = new ParallelOnlineLogisticRegression(this.num_categories,
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
    POLRMasterNode pmn = new POLRMasterNode();
    ApplicationMaster<ParameterVectorUpdatable> am = new ApplicationMaster<ParameterVectorUpdatable>(
        pmn, ParameterVectorUpdatable.class);
    
    ToolRunner.run(am, args);
  }
  
}
