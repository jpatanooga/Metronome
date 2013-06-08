package tv.floe.metronome.clustering.kmeans.ir;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;

import com.cloudera.iterativereduce.ComputableMaster;
import tv.floe.metronome.clustering.kmeans.Means;

public class KMeansMaster implements ComputableMaster<UpdateableMeans> {

	private static final Log LOG = LogFactory.getLog(KMeansMaster.class);
	private UpdateableMeans um;
	
	@Override
	public void setup(Configuration c) {
		um = new UpdateableMeans();
	}

	@Override
	public void complete(DataOutputStream out) throws IOException {
		 out.write(um.toString().getBytes());
	}

	@Override
	public UpdateableMeans compute(Collection<UpdateableMeans> workerUpdates,
			Collection<UpdateableMeans> masterUpdates) {
		Means means = um.get();
		means.reset();
		
		for(UpdateableMeans m : workerUpdates) {
			System.out.println(m);
			means.merge(m.get());
		}
		
		um.set(means);
		LOG.info("Calculated new means: " + means);
		return um;
	}

	@Override
	public UpdateableMeans getResults() {
		return um;
	}

}
