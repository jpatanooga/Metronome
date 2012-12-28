package tv.floe.metronome.linearregression;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.io.Writable;

//import com.cloudera.knittingboar.sgd.POLRModelParameters;
//import com.cloudera.knittingboar.sgd.ParallelOnlineLogisticRegression;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;

public class ModelParameters implements Writable {
	  
	  private String targetVariable;
	  private Map<String,String> typeMap;
	  private int numFeatures;
	  private boolean useBias;
	  private int maxTargetCategories;
	  private List<String> targetCategories;
	  private double lambda;
	  private double learningRate;
	  
	  private ParallelOnlineLinearRegression polr;
	  
	  /**
	   * Saves a model to an output stream.
	   * 
	   * @throws Exception
	   */
	  public void saveTo(OutputStream out) throws Exception {
	    if (polr != null) {
	      polr.close();
	    } else {
	      System.out.println("Model Save >>> polr is null! [ERR]");
	    }
	    if (null == this.targetCategories) {
	      System.out.println("targetCategories is null!");
	      throw new Exception("targetCategories is null!");
	    }
	    
	    write(new DataOutputStream(out));
	    
	  }
	  
	  /**
	   * Reads a model from a stream.
	   */
	  public static ModelParameters loadFrom(InputStream in) throws IOException {
		  ModelParameters result = new ModelParameters();
	    result.readFields(new DataInputStream(in));
	    return result;
	  }
	  
	  /**
	   * Reads a model from a file.
	   * 
	   * @throws IOException
	   *           If there is an error opening or closing the file.
	   */
	  public static ModelParameters loadFrom(File in) throws IOException {
	    InputStream input = new FileInputStream(in);
	    try {
	      return loadFrom(input);
	    } finally {
	      Closeables.closeQuietly(input);
	    }
	  }
	  
	  /**
	   * Write member data out to the output stream
	   */
	  @Override
	  public void write(DataOutput out) throws IOException {
	    out.writeUTF(targetVariable);
	    out.writeInt(typeMap.size());
	    for (Map.Entry<String,String> entry : typeMap.entrySet()) {
	      out.writeUTF(entry.getKey());
	      out.writeUTF(entry.getValue());
	    }
	    out.writeInt(numFeatures);
	    out.writeBoolean(useBias);
	    out.writeInt(maxTargetCategories);
	    out.writeInt(targetCategories.size());
	    for (String category : targetCategories) {
	      out.writeUTF(category);
	    }
	    out.writeDouble(lambda);
	    System.out.println("write lambda: " + lambda);
	    out.writeDouble(learningRate);
	    // skip csv
	    polr.write(out);
	  }
	  
	  /**
	   * 
	   * Read appropriate fields from the InputStream
	   * 
	   */
	  @Override
	  public void readFields(DataInput in) throws IOException {
	    targetVariable = in.readUTF();
	    int typeMapSize = in.readInt();
	    typeMap = Maps.newHashMapWithExpectedSize(typeMapSize);
	    for (int i = 0; i < typeMapSize; i++) {
	      String key = in.readUTF();
	      String value = in.readUTF();
	      typeMap.put(key, value);
	    }
	    numFeatures = in.readInt();
	    useBias = in.readBoolean();
	    maxTargetCategories = in.readInt();
	    int targetCategoriesSize = in.readInt();
	    targetCategories = Lists.newArrayListWithCapacity(targetCategoriesSize);
	    for (int i = 0; i < targetCategoriesSize; i++) {
	      targetCategories.add(in.readUTF());
	    }
	    lambda = in.readDouble();
	    learningRate = in.readDouble();
	    System.out.println("read lambda: " + lambda);
	    // csv = null;
	    polr = new ParallelOnlineLinearRegression();
	    polr.readFields(in);
	  }
	  
	  /**
	   * Sets the types of the predictors. This will later be used when reading CSV
	   * data.
	   * 
	   * If you don't use the CSV data and convert to vectors on your own, you don't
	   * need to call this.
	   * 
	   * @param predictorList
	   *          The list of variable names.
	   * @param typeList
	   *          The list of types in the format preferred by CsvRecordFactory.
	   */
	  public void setTypeMap(Iterable<String> predictorList, List<String> typeList) {
	    Preconditions.checkArgument(!typeList.isEmpty(),
	        "Must have at least one type specifier");
	    typeMap = Maps.newHashMap();
	    Iterator<String> iTypes = typeList.iterator();
	    String lastType = null;
	    for (Object x : predictorList) {
	      // type list can be short .. we just repeat last spec
	      if (iTypes.hasNext()) {
	        lastType = iTypes.next();
	      }
	      
	      typeMap.put(x.toString(), lastType);
	    }
	  }
	  
	  /**
	   * Sets the target variable. If you don't use the CSV record factory, then
	   * this is irrelevant.
	   * 
	   * @param targetVariable
	   *          The name of the target variable.
	   */
	  public void setTargetVariable(String targetVariable) {
	    this.targetVariable = targetVariable;
	  }
	  
	  /**
	   * Sets the number of target categories to be considered.
	   * 
	   * @param maxTargetCategories
	   *          The number of target categories.
	   */
	  public void setMaxTargetCategories(int maxTargetCategories) {
	    this.maxTargetCategories = maxTargetCategories;
	  }
	  
	  public void setNumFeatures(int numFeatures) {
	    this.numFeatures = numFeatures;
	  }
	  
	  public void setTargetCategories(List<String> targetCategories) {
	    this.targetCategories = targetCategories;
	    maxTargetCategories = targetCategories.size();
	  }
	  
	  public List<String> getTargetCategories() {
	    return this.targetCategories;
	  }
	  
	  public void setUseBias(boolean useBias) {
	    this.useBias = useBias;
	  }
	  
	  public boolean useBias() {
	    return useBias;
	  }
	  
	  public String getTargetVariable() {
	    return targetVariable;
	  }
	  
	  public Map<String,String> getTypeMap() {
	    return typeMap;
	  }
	  
	  public void setTypeMap(Map<String,String> map) {
	    this.typeMap = map;
	  }
	  
	  public int getNumFeatures() {
	    return numFeatures;
	  }
	  
	  public int getMaxTargetCategories() {
	    return maxTargetCategories;
	  }
	  
	  public double getLambda() {
	    return lambda;
	  }
	  
	  public void setLambda(double lambda) {
	    this.lambda = lambda;
	  }
	  
	  public double getLearningRate() {
	    return learningRate;
	  }
	  
	  public void setLearningRate(double learningRate) {
	    this.learningRate = learningRate;
	  }
	  
	  public void setPOLR(ParallelOnlineLinearRegression plr) {
	    this.polr = plr;
	  }
	  
	  public ParallelOnlineLinearRegression getPOLR() {
	    this.polr.lambda(lambda);
	    this.polr.learningRate(learningRate);
	    return this.polr;
	  }
	  
	  public void Debug() throws IOException {
	    
	    System.out.println("# POLRModelParams ------------ Debug ---------");
	    
	    System.out.println("> Num Categories: " + this.maxTargetCategories);
	    System.out.println("> TypeMapSize: " + typeMap.size());
	    for (Map.Entry<String,String> entry : typeMap.entrySet()) {
	      System.out.println(">>\t Key: " + entry.getKey().toString());
	      System.out.println(">>\t Val: " + entry.getValue().toString());
	    }
	    System.out.println("> numFeatures: " + numFeatures);
	    System.out.println("> useBias: " + useBias);
	    System.out.println("> maxTargetCategories: " + maxTargetCategories);
	    System.out.println("> targetCategories.size(): " + targetCategories.size());
	    for (String category : targetCategories) {
	      System.out.println(">>\t category: " + category);
	    }
	    System.out.println("> lambda: " + lambda);
	    System.out.println("> learningRate: " + learningRate);
	  }
	  
}
