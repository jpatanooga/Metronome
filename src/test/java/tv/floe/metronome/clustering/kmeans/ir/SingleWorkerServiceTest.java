package tv.floe.metronome.clustering.kmeans.ir;

import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import com.cloudera.iterativereduce.ComputableMaster;
import com.cloudera.iterativereduce.ComputableWorker;
import com.cloudera.iterativereduce.Utils;
import com.cloudera.iterativereduce.io.TextRecordParser;
import com.cloudera.iterativereduce.yarn.appmaster.ApplicationMasterService;
import com.cloudera.iterativereduce.yarn.appworker.ApplicationWorkerService;
import com.cloudera.iterativereduce.yarn.avro.generated.FileSplit;
import com.cloudera.iterativereduce.yarn.avro.generated.StartupConfiguration;
import com.cloudera.iterativereduce.yarn.avro.generated.WorkerId;

public class SingleWorkerServiceTest {

	InetSocketAddress masterAddress;
	ExecutorService pool;
/*
	private ApplicationMasterService<UpdateableMeans> masterService;
	private Future<Integer> master;
	private ComputableMaster<UpdateableMeans> computableMaster;

	private ApplicationWorkerService<UpdateableMeans> workerService;
	private ComputableWorker<UpdateableMeans> computableWorker;

	private Random random;

	private Point generatePoint(Point p, double sigma) {
		double [] d = new double[p.dimensionality()];
		for(int i = 0; i < d.length; i++) {
			d[i] = p.get(i) + sigma * random.nextGaussian();
		}
		return new Point(d);
	}

	private List<Point> generateGaussianPoints(Point p, double sigma, int points) {
		List<Point> result = new ArrayList<Point>();
		for(int i = 0; i < points; i++) {
			result.add(generatePoint(p, sigma));
		}
		return result;
	}

	@Before
	public void setUp() throws Exception {
		masterAddress = new InetSocketAddress(9999);
		pool = Executors.newFixedThreadPool(2);

		setUpMaster();
		Thread.sleep(500);
	}

	@Before
	public void setUpFile() throws Exception {
		random = new Random(0);
		Configuration conf = new Configuration();
		FileSystem localFs = FileSystem.getLocal(conf);
		Path testDir = new Path("testData");
		Path inputFile = new Path(testDir, "testWorkerService.txt");
		Writer writer = new OutputStreamWriter(localFs.create(inputFile, true));
		for(Point point : generateGaussianPoints(new Point(0,0), 0.2, 20)){
			writer.write(point.get(0) + "," + point.get(1) + "\n");
		}
		for(Point point : generateGaussianPoints(new Point(1,1), 0.2, 20)){
			writer.write(point.get(0) + "," + point.get(1) + "\n");
		}
		for(Point point : generateGaussianPoints(new Point(2,2), 0.2, 20)){
			writer.write(point.get(0) + "," + point.get(1) + "\n");
		}
		writer.close();
	}

	@After
	public void cleanup() {
		pool.shutdown();
	}

	public void setUpMaster() throws Exception {
		FileSplit split = FileSplit.newBuilder()
				.setPath("testData/testWorkerService.txt").setOffset(0).setLength(Long.MAX_VALUE)
				.build();

		StartupConfiguration conf = StartupConfiguration.newBuilder()
				.setSplit(split).setBatchSize(200).setIterations(10).setOther(null)
				.build();

		HashMap<WorkerId, StartupConfiguration> workers = new HashMap<WorkerId, StartupConfiguration>();
		workers.put(Utils.createWorkerId("worker1"), conf);

		computableMaster = new KMeansMaster();
		masterService = new ApplicationMasterService<UpdateableMeans>(masterAddress,
				workers, computableMaster, UpdateableMeans.class);

		master = pool.submit(masterService);
	}

	@Test
	public void testWorkerService() throws Exception {
		TextRecordParser<UpdateableMeans> parser = new TextRecordParser<UpdateableMeans>();
		computableWorker = new KMeansWorker();
		workerService = new ApplicationWorkerService<UpdateableMeans>(
				"worker1", masterAddress, parser, computableWorker, UpdateableMeans.class);

		workerService.run();
		
		Means means = computableMaster.getResults().get();
		System.out.println("Means:");
		for(Mean mean : means) {
			System.out.println(mean.toPoint());
		}
		master.get();
	}

	public static void main(String[] args) throws Exception {
		SingleWorkerServiceTest tsws = new SingleWorkerServiceTest();
		tsws.setUp();
		tsws.testWorkerService();
	}
	*/
}
