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
 * @author jpatterson
 * 
 */
public class RCV1RecordFactory implements RecordFactory {
  
  public static final int FEATURES = 10000;
  ConstantValueEncoder encoder = null;
  
  public RCV1RecordFactory() {
    
    this.encoder = new ConstantValueEncoder("body_values");
    
  }
  
  public static void ScanFile(String file, int debug_break_cnt)
      throws IOException {
    
    //ConstantValueEncoder encoder_test = new ConstantValueEncoder("test");
    
    BufferedReader reader = null;
    // Collection<String> words
    int line_count = 0;
    
    Multiset<String> class_count = ConcurrentHashMultiset.create();
    Multiset<String> namespaces = ConcurrentHashMultiset.create();
    
    try {
      // System.out.println( newsgroup );
      reader = new BufferedReader(new FileReader(file));
      
      String line = reader.readLine();
      
      while (line != null && line.length() > 0) {
        
        // shard_writer.write(line + "\n");
        // out += line;
        
        String[] parts = line.split(" ");
        
        // System.out.println( "Class: " + parts[0] );
        
        class_count.add(parts[0]);
        namespaces.add(parts[1]);
        
        line = reader.readLine();
        line_count++;
        
        Vector v = new RandomAccessSparseVector(FEATURES);
        
        for (int x = 2; x < parts.length; x++) {
          // encoder_test.addToVector(parts[x], v);
          // System.out.println( parts[x] );
          String[] feature = parts[x].split(":");
          int index = Integer.parseInt(feature[0]) % FEATURES;
          double val = Double.parseDouble(feature[1]);
          
          // System.out.println( feature[1] + " = " + val );
          
          if (index < FEATURES) {
            v.set(index, val);
          } else {
            
            System.out.println("Could Hash: " + index + " to "
                + (index % FEATURES));
            
          }
          
        }
        
//        Utils.PrintVectorSectionNonZero(v, 10);
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
      
      /*
       * TokenStream ts = analyzer.tokenStream("text", reader);
       * ts.addAttribute(CharTermAttribute.class);
       * 
       * // for each word in the stream, minus non-word stuff, add word to
       * collection while (ts.incrementToken()) { String s =
       * ts.getAttribute(CharTermAttribute.class).toString();
       * //System.out.print( " " + s ); //words.add(s); out += s + " "; }
       */

    } finally {
      reader.close();
    }
    
    // return out + "\n";
    
  }
  
  // doesnt really do anything in a 2 class dataset
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
  public double processLineNew(String line, Vector v) throws Exception {
    
    // p.269 ---------------------------------------------------------
    // Map<String, Set<Integer>> traceDictionary = new TreeMap<String,
    // Set<Integer>>();
    
    double actual = 0;
    
    String[] parts = line.split(" ");
    
    //actual = Integer.parseInt(parts[0]);
    actual = Double.parseDouble(parts[0]);
    
    // dont know what to do the the "namespace" "f"
    
    for (int x = 2; x < parts.length; x++) {
      
      String[] feature = parts[x].split(":");
      int index = Integer.parseInt(feature[0]) % FEATURES;
      double val = Double.parseDouble(feature[1]);
      
      if (index < FEATURES) {
        v.set(index, val);
      } else {
        
        System.out
            .println("Could Hash: " + index + " to " + (index % FEATURES));
        
      }
      
    }
    
    // System.out.println("\nEOL\n");
    
    return actual;
  }
  
  @Override
  public List<String> getTargetCategories() {
    
    List<String> out = new ArrayList<String>();
    
    // for ( int x = 0; x < this.newsGroups.size(); x++ ) {
    
    // System.out.println( x + "" + this.newsGroups.values().get(x) );
    out.add("0");
    out.add("1");
    
    // }
    
    return out;
    
  }

@Override
public int processLine(String line, Vector featureVector) throws Exception {
	// TODO Auto-generated method stub
	return 0;
}
  
}
