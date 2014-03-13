package tv.floe.metronome.deeplearning.dbn.iterativereduce;

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;

import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;


import com.cloudera.iterativereduce.ComputableWorker;
import com.cloudera.iterativereduce.io.RecordParser;

/**
 * Base IterativeReduce worker node
 * 
 * TODO
 * - setup a base DBN instance to run locally
 * 
 * @author josh
 *
 */
public class WorkerNode implements ComputableWorker<ParameterVectorUpdateable> {
	
	private static final Log LOG = LogFactory.getLog(WorkerNode.class);
	
	DeepBeliefNetwork dbn = null;
	
	@Override
	public boolean IncrementIteration() {
		// TODO Auto-generated method stub
		return false;
	}
	
	/**
	 * Run a training pass on the DBN
	 * 
	 */
	@Override
	public ParameterVectorUpdateable compute() {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public ParameterVectorUpdateable compute(List<ParameterVectorUpdateable> arg0) {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public ParameterVectorUpdateable getResults() {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public void setRecordParser(RecordParser arg0) {
		// TODO Auto-generated method stub
		
	}
	
	/**
	 * Setup the local DBN instance based on conf params
	 * 
	 */
	@Override
	public void setup(Configuration arg0) {
		// TODO Auto-generated method stub
		
	}
	
	/**
	 * Collect the update from the master node and apply it to the local 
	 * parameter vector
	 * 
	 */
	@Override
	public void update(ParameterVectorUpdateable arg0) {
		// TODO Auto-generated method stub
		
	}


}
