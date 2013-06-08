package tv.floe.metronome.io.records;


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



import java.util.List;
import java.util.Map;
import java.util.Set;

//import org.apache.mahout.classifier.sgd.RecordFactory;
import org.apache.mahout.math.Vector;

/**
 * Base interface for Knitting Boar's vectorization system
 * 
 * @author jpatterson
 * 
 */
public interface RecordFactory {
  
  public static String TWENTYNEWSGROUPS_RECORDFACTORY = "com.cloudera.knittingboar.records.TwentyNewsgroupsRecordFactory";
  public static String RCV1_RECORDFACTORY = "com.cloudera.knittingboar.records.RCV1RecordFactory";
  public static String CSV_RECORDFACTORY = "com.cloudera.knittingboar.records.CSVRecordFactory";
  
  public int processLine(String line, Vector featureVector) throws Exception;
  public double processLineAlt(String line, Vector v) throws Exception;
  
  public String GetClassnameByID(int id);
  
  // Map<String, Set<Integer>> getTraceDictionary();
  
  // RecordFactory includeBiasTerm(boolean useBias);
  
  public List<String> getTargetCategories();
  
  public int getFeatureVectorSize();
  
  // void firstLine(String line);
  
}
