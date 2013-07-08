package tv.floe.metronome.classification.neuralnetworks.iterativereduce;

import java.util.List;

import org.apache.hadoop.conf.Configuration;

import tv.floe.metronome.linearregression.iterativereduce.NodeBase;

import com.cloudera.iterativereduce.ComputableWorker;
import com.cloudera.iterativereduce.io.RecordParser;

public class WorkerNode extends NodeBase implements ComputableWorker<WeightsUpdateable> {

	@Override
	public boolean IncrementIteration() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public WeightsUpdateable compute() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public WeightsUpdateable compute(List<WeightsUpdateable> arg0) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public WeightsUpdateable getResults() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setRecordParser(RecordParser arg0) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setup(Configuration arg0) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void update(WeightsUpdateable arg0) {
		// TODO Auto-generated method stub
		
	}

}
