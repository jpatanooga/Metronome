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


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

import org.apache.mahout.math.RandomAccessSparseVector;

//import com.cloudera.knittingboar.utils.Utils;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;

/**
 * RecordFactory for
 * https://github.com/JohnLangford/vowpal_wabbit/wiki/Rcv1-example
 * 
 * TODO: handle the case of the intercept as the 0 index slot
 * 
 * @author jpatterson
 * 
 */
public class RCV1RecordFactory implements RecordFactory {
  
  public static final int FEATURES = 10000;
  
  public RCV1RecordFactory() {
    
    
  }
  
  public static void ScanFile(String file, int debug_break_cnt)
      throws IOException {
    
    
    BufferedReader reader = null;
    int line_count = 0;
    
    Multiset<String> class_count = ConcurrentHashMultiset.create();
    Multiset<String> namespaces = ConcurrentHashMultiset.create();
    
    try {
      reader = new BufferedReader(new FileReader(file));
      
      String line = reader.readLine();
      
      while (line != null && line.length() > 0) {
        
        
        String[] parts = line.split(" ");
        
        
        class_count.add(parts[0]);
        namespaces.add(parts[1]);
        
        line = reader.readLine();
        line_count++;
        
        Vector v = new RandomAccessSparseVector(FEATURES);
        
        for (int x = 2; x < parts.length; x++) {
          String[] feature = parts[x].split(":");
          int index = Integer.parseInt(feature[0]) % FEATURES;
          double val = Double.parseDouble(feature[1]);
          
          System.out.println( feature[1] + " = " + val );
          
          if (index < FEATURES) {
            v.set(index, val);
          } else {
            
            System.out.println("Could Hash: " + index + " to "
                + (index % FEATURES));
            
          }
          
        }
        
        System.out.println("###");
        
        if (line_count > debug_break_cnt) {
          break;
        }
        
      }
      
      System.out.println("Total Rec Count: " + line_count);
      
      System.out.println("-------------------- ");
      
      System.out.println("Classes");
      for (String word : class_count.elementSet()) {
        System.out.println("Class " + word + ": " + class_count.count(word)
            + " ");
      }
      
      System.out.println("-------------------- ");
      
      System.out.println("NameSpaces:");
      for (String word : namespaces.elementSet()) {
        System.out.println("Namespace " + word + ": " + namespaces.count(word)
            + " ");
      }
      

    } finally {
      reader.close();
    }
    
    
  }
  
  @Override
  public String GetClassnameByID(int id) {
    return String.valueOf(id); // this.newsGroups.values().get(id);
  }
  
  /**
   * Processes single line of input into: - target variable - Feature vector
   * 
   * Right now our hash function is simply "modulo"
   * 
   * @throws Exception
   */
  @Override
  public double processLineAlt(String line, Vector v) throws Exception {
    
    double actual = 0;
    
    String[] parts = line.split(" ");
    
    actual = Double.parseDouble(parts[0]);
    
    // dont know what to do the the "namespace" "f"
    v.set(0, 1.0);
    
    for (int x = 2; x < parts.length; x++) {
      
      String[] feature = parts[x].split(":");
      int index = (Integer.parseInt(feature[0]) + 1) % FEATURES;
      double val = Double.parseDouble(feature[1]);
      
      if (index < FEATURES) {
        v.set(index, val);
      } else {
        
        System.out
            .println("Could Hash: " + index + " to " + (index % FEATURES));
        
      }
      
    }
    
    return actual;
  }
  
  @Override
  public List<String> getTargetCategories() {
    
    List<String> out = new ArrayList<String>();
    
    out.add("0");
    out.add("1");
    
    return out;
    
  }

	@Override
	public int processLine(String line, Vector featureVector) throws Exception {
		return 0;
	}

	@Override
	public int getFeatureVectorSize() {
		return this.FEATURES;
	}
	
	@Override
	public void parseSchema() {
		// implied schema, does nothing here
	}
	    
	@Override
	public int getInputVectorSize() {
		return this.FEATURES;
	}
	  
	@Override
	public int getOutputVectorSize() {
		return 1;
	}
	
	// using the older method above, but wired this for the cached vector reader
	@Override
	public void vectorizeLine(String line, Vector v_in, Vector v_out) throws Exception {
		double out = this.processLineAlt( line, v_in );
		v_out.set(0, out);
	}
  
}
