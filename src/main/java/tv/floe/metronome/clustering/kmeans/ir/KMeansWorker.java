package tv.floe.metronome.clustering.kmeans.ir;

import java.io.IOException;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;

import com.cloudera.iterativereduce.ComputableWorker;
import com.cloudera.iterativereduce.io.RecordParser;
import com.cloudera.iterativereduce.io.TextRecordParser;
import tv.floe.metronome.clustering.kmeans.KMeansPartition;
import tv.floe.metronome.clustering.kmeans.MutablePoint;

public class KMeansWorker implements ComputableWorker<UpdateableMeans> {

	private static final Log LOG = LogFactory.getLog(KMeansWorker.class);

	private UpdateableMeans um = new UpdateableMeans();
	@SuppressWarnings("rawtypes")
	private TextRecordParser rp;
	private Text t = new Text();

	private KMeansPartition partition;

	@Override
	public void setup(Configuration c) {
//		partition = new KMeansPartition(c.getInt("k", -1));
		// TODO Make this not hardcoded
		partition = new KMeansPartition(3);
	}

	@Override
	public UpdateableMeans compute(List<UpdateableMeans> records) {
		throw new UnsupportedOperationException();
	}

	@Override
	public UpdateableMeans compute() {
		int points = 0;
		MutablePoint mp = new MutablePoint();
		try { 
			while (rp.next(t)) {
				mp.parse(t.toString());
				partition.addPoint(mp);
				points++;
			}
		} catch (IOException ex) {
			LOG.warn(ex);
		}
		LOG.info("I iterated over " + points + " points.");
		um.set(partition.getUpdatedMeans());
		return um;
	}

	@SuppressWarnings("rawtypes")
	@Override
	public void setRecordParser(RecordParser r) {
		rp = (TextRecordParser) r;
	}

	@Override
	public UpdateableMeans getResults() {
		return um;
	}

	@Override
	public void update(UpdateableMeans t) {
		um = t;
		partition.setMeans(um.get());
	}

	@Override
	public boolean IncrementIteration() {
		return false;
	}

}
