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

package tv.floe.metronome.classification.logisticregression.metrics;

/**
 * this is the class we'll use to report worker node perf to master node
 * 
 * @author jpatterson
 *
 */
public class POLRMetrics {
  
  public String WorkerNodeIPAddress = null;
  public String WorkerNodeInputDataSplit = null;
  
  public long AvgBatchProecssingTimeInMS = 0;
  
  public long TotalInputProcessingTimeInMS = 0;
  public long TotalRecordsProcessed = 0;
  
  public double AvgLogLikelihood = 0.0;
  public double AvgCorrect = 0.0;

}
