package tv.floe.metronome.clustering.kmeans.ir;

import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
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

public class MultipleWorkerServicesTest {

	InetSocketAddress masterAddress;
	ExecutorService pool;
/*
	private ApplicationMasterService<UpdateableMeans> masterService;
	private Future<Integer> master;
	private ComputableMaster<UpdateableMeans> computableMaster;

	private ArrayList<ApplicationWorkerService<UpdateableMeans>> workerServices = new ArrayList<ApplicationWorkerService<UpdateableMeans>>();
	private ArrayList<Future<Integer>> workers = new ArrayList<Future<Integer>>();
	private ArrayList<ComputableWorker<UpdateableMeans>> computableWorkers = new ArrayList<ComputableWorker<UpdateableMeans>>();

	private Random random;
	
	@Before
	public void setUp() throws Exception {
		masterAddress = new InetSocketAddress(9999);
		pool = Executors.newFixedThreadPool(4);

		random = new Random(0);
		setUpFiles();
		setUpMaster();

		// Wait for master thread to do its stuff...
		Thread.sleep(1000);

		setUpWorker("worker1");
		setUpWorker("worker2");
		setUpWorker("worker3");
	}

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


	public void setUpFiles() throws Exception {
		Configuration conf = new Configuration();
		FileSystem localFs = FileSystem.getLocal(conf);
		Path testDir = new Path("testData");
		for(int i = 0; i < 3; i++) {
			Path inputFile = new Path(testDir, "testWorkerService" + i + ".txt");
			Writer writer = new OutputStreamWriter(localFs.create(inputFile, true));
			for(Point point : generateGaussianPoints(new Point(0,0), 0.2, 5)){
				writer.write(point.get(0) + "," + point.get(1) + "\n");
			}
			for(Point point : generateGaussianPoints(new Point(1,1), 0.2, 5)){
				writer.write(point.get(0) + "," + point.get(1) + "\n");
			}
			for(Point point : generateGaussianPoints(new Point(2,2), 0.2, 5)){
				writer.write(point.get(0) + "," + point.get(1) + "\n");
			}
			writer.close();
		}
	}

	public void setUpMaster() throws Exception {
		FileSplit split0 = FileSplit.newBuilder()
				.setPath("testData/testWorkerService0.txt").setOffset(0).setLength(Long.MAX_VALUE)
				.build();
		FileSplit split1 = FileSplit.newBuilder()
				.setPath("testData/testWorkerService1.txt").setOffset(0).setLength(Long.MAX_VALUE)
				.build();
		FileSplit split2 = FileSplit.newBuilder()
				.setPath("testData/testWorkerService2.txt").setOffset(0).setLength(Long.MAX_VALUE)
				.build();

		StartupConfiguration conf0 = StartupConfiguration.newBuilder()
				.setSplit(split0).setBatchSize(0).setIterations(20).setOther(null)
				.build();
		StartupConfiguration conf1 = StartupConfiguration.newBuilder()
				.setSplit(split1).setBatchSize(0).setIterations(20).setOther(null)
				.build();
		StartupConfiguration conf2 = StartupConfiguration.newBuilder()
				.setSplit(split2).setBatchSize(0).setIterations(20).setOther(null)
				.build();

		HashMap<WorkerId, StartupConfiguration> workers = new HashMap<WorkerId, StartupConfiguration>();
		workers.put(Utils.createWorkerId("worker1"), conf0);
		workers.put(Utils.createWorkerId("worker2"), conf1);
		workers.put(Utils.createWorkerId("worker3"), conf2);

		computableMaster = new KMeansMaster();
		masterService = new ApplicationMasterService<UpdateableMeans>(masterAddress,
				workers, computableMaster, UpdateableMeans.class);

		master = pool.submit(masterService);
	}

	private void setUpWorker(String name) {
		TextRecordParser parser = new TextRecordParser();
		ComputableWorker<UpdateableMeans> computableWorker = new KMeansWorker();
		final ApplicationWorkerService<UpdateableMeans> workerService = new ApplicationWorkerService<UpdateableMeans>(
				name, masterAddress, parser, computableWorker, UpdateableMeans.class);

		Future<Integer> worker = pool.submit(new Callable<Integer>() {
			public Integer call() {
				return workerService.run();
			}
		});

		computableWorkers.add(computableWorker);
		workerServices.add(workerService);
		workers.add(worker);
	}

	@Test
	public void testWorkerService() throws Exception {
		workers.get(0).get();
		workers.get(1).get();
		workers.get(2).get();
		master.get();
	
		Means means = computableMaster.getResults().get();
		System.out.println("Means:");
		for(Mean mean : means) {
			System.out.println(mean.toPoint());
		}
	}
	*/
}
