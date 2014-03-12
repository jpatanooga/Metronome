package tv.floe.metronome.deeplearning.dbn.iterativereduce;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;

import org.apache.hadoop.conf.Configuration;

import com.cloudera.iterativereduce.ComputableMaster;


public class MasterNode implements ComputableMaster<ParameterVectorUpdateable> {

	@Override
	public void complete(DataOutputStream arg0) throws IOException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public ParameterVectorUpdateable compute(
			Collection<ParameterVectorUpdateable> arg0,
			Collection<ParameterVectorUpdateable> arg1) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ParameterVectorUpdateable getResults() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setup(Configuration arg0) {
		// TODO Auto-generated method stub
		
	}

}
