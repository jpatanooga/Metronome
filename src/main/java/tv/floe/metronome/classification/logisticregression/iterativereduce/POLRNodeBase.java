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

import org.apache.hadoop.conf.Configuration;

/**
 * Base class for IR-KnittingBoar nodes
 * 
 * @author jpatterson
 * 
 */
public class POLRNodeBase {
  
  protected Configuration conf = null;
  protected int num_categories = 2;
  protected int FeatureVectorSize = -1;
//  protected int BatchSize = 200;
  protected double Lambda = 1.0e-4;
  protected double LearningRate = 10;
  
  String LocalInputSplitPath = "";
  String PredictorLabelNames = "";
  String PredictorVariableTypes = "";
  protected String TargetVariableName = "";
  protected String ColumnHeaderNames = "";
  protected int NumberIterations = 1;
  
  protected int LocalBatchCountForIteration = 0;
  protected int GlobalBatchCountForIteration = 0;
  
  protected String RecordFactoryClassname = "";
  
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
  
  public Configuration getConf() {
    return this.conf;
  }
  
}
