package tv.floe.metronome.linearregression.iterativereduce;

import org.apache.hadoop.conf.Configuration;

public class NodeBase {
	  
	  protected Configuration conf = null;
	  protected int FeatureVectorSize = -1;
	  protected double Lambda = 1.0e-4;
	  protected double LearningRate = 10;
	  
	  String PredictorLabelNames = "";
	  String PredictorVariableTypes = "";
	  protected String TargetVariableName = "";
	  protected String ColumnHeaderNames = "";
	  protected int NumberIterations = 1;
	  
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

}
