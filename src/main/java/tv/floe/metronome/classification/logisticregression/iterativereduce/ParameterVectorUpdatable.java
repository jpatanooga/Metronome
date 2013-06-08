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

import java.io.IOException;
import java.nio.ByteBuffer;

import org.apache.mahout.math.Matrix;


//import com.cloudera.knittingboar.sgd.GradientBuffer;
import com.cloudera.iterativereduce.Updateable;

public class ParameterVectorUpdatable implements
    Updateable<ParameterVector> {
  
  
  ParameterVector param_msg = null;
  
  public ParameterVectorUpdatable() {}
  
  public ParameterVectorUpdatable(ParameterVector g) {
    this.param_msg = g;
  }
  
  @Override
  public void fromBytes(ByteBuffer b) {
    
    b.rewind();
    
    // System.out.println( " > ParameterVectorGradient::fromBytes > b: " +
    // b.array().length + ", remaining: " + b.remaining() );
    
    try {
      this.param_msg = new ParameterVector();
      this.param_msg.Deserialize(b.array());
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }
  
  @Override
  public ParameterVector get() {
    // TODO Auto-generated method stub
    return this.param_msg;
  }
  
  @Override
  public void set(ParameterVector t) {
    // TODO Auto-generated method stub
    this.param_msg = t;
  }
  
  @Override
  public ByteBuffer toBytes() {
    // TODO Auto-generated method stub
    byte[] bytes = null;
    try {
      bytes = this.param_msg.Serialize();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    
    // ByteBuffer buf = ByteBuffer.allocate(bytes.length);
    // buf.put(bytes);
    ByteBuffer buf = ByteBuffer.wrap(bytes);
    
    return buf;
  }
  
  @Override
  public void fromString(String s) {
  // TODO Auto-generated method stub
  
  }
/*
  @Override
  public int getGlobalBatchNumber() {
    // TODO Auto-generated method stub
    return 0;
  }

  @Override
  public int getGlobalIterationNumber() {
    // TODO Auto-generated method stub
    return 0;
  }

  @Override
  public void setIterationState(int arg0, int arg1) {
    // TODO Auto-generated method stub
    
  }
*/  
}
