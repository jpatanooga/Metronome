package tv.floe.metronome.classification.neuralnetworks.iterativereduce;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;

import org.apache.hadoop.conf.Configuration;

import tv.floe.metronome.linearregression.iterativereduce.NodeBase;

import com.cloudera.iterativereduce.ComputableMaster;

public class MasterNode  extends NodeBase implements ComputableMaster<WeightsUpdateable> {

	@Override
	public void complete(DataOutputStream arg0) throws IOException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public WeightsUpdateable compute(Collection<WeightsUpdateable> arg0,
			Collection<WeightsUpdateable> arg1) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public WeightsUpdateable getResults() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setup(Configuration arg0) {
		// TODO Auto-generated method stub
		
	}



}
