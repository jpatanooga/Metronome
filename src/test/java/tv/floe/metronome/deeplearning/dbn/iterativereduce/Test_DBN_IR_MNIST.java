package tv.floe.metronome.deeplearning.dbn.iterativereduce;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.junit.Test;

import com.cloudera.iterativereduce.io.TextRecordParser;

import tv.floe.metronome.irunit.IRUnitDriver;
import tv.floe.metronome.deeplearning.datasets.DataSet;
import tv.floe.metronome.deeplearning.datasets.iterator.impl.MnistHDFSDataSetIterator;
import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;
import tv.floe.metronome.deeplearning.dbn.iterativereduce.*;
import tv.floe.metronome.deeplearning.dbn.model.evaluation.ModelTester;

public class Test_DBN_IR_MNIST {



	  private static JobConf defaultConf = new JobConf();
	  private static FileSystem localFs = null; 
	  static {
	    try {
	      defaultConf.set("fs.defaultFS", "file:///");
	      localFs = FileSystem.getLocal(defaultConf);
	    } catch (IOException e) {
	      throw new RuntimeException("init failure", e);
	    }
	  }
	
	
	private InputSplit[] generateDebugSplits(Path input_path, JobConf job) {

		long block_size = localFs.getDefaultBlockSize();

		System.out.println("default block size: " + (block_size / 1024 / 1024)
				+ "MB");

		// ---- set where we'll read the input files from -------------
		FileInputFormat.setInputPaths(job, input_path);

		// try splitting the file in a variety of sizes
		TextInputFormat format = new TextInputFormat();
		format.configure(job);

		int numSplits = 1;

		InputSplit[] splits = null;

		try {
			splits = format.getSplits(job, numSplits);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return splits;

	}		
	
	public DataSet setupHDFSDataset( String vectors_filename ) throws IOException {
		
		
		int batchSize = 20;
		int totalNumExamples = 20;

		// setup splits ala HDFS style -------------------
		
	    JobConf job = new JobConf(defaultConf);
	    
	    Path workDir = new Path( vectors_filename );
		
		
	    InputSplit[] splits = generateDebugSplits(workDir, job);
	    
	    System.out.println( "> splits: " + splits[0].toString() );

	    
	    TextRecordParser txt_reader = new TextRecordParser();

	    long len = Integer.parseInt(splits[0].toString().split(":")[2]
	        .split("\\+")[1]);

	    txt_reader.setFile(splits[0].toString().split(":")[1], 0, len);		
		
				

	    
		MnistHDFSDataSetIterator hdfs_fetcher = new MnistHDFSDataSetIterator( batchSize, totalNumExamples, txt_reader );
		DataSet hdfs_recordBatch = hdfs_fetcher.next();
		
		return hdfs_recordBatch;
	}
	
	
	@Test
	public void testIR() throws Exception {
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/unit_tests/dbn/mnist/app.unit_test.dbn.mnist.properties");
		polr_ir.Setup();

		polr_ir.SimulateRun();

	}

	
	@Test
	public void testIR_DBN_MNIST_OneWorker_TwoLabels() throws Exception {
		
		String yarn_props_file = "src/test/resources/run_profiles/unit_tests/dbn/mnist/app.unit_test.dbn.mnist.two_labels.properties";
		
		IRUnitDriver polr_ir = new IRUnitDriver( yarn_props_file );
		polr_ir.Setup();

		polr_ir.SimulateRun();

		ModelTester.evaluateModel( yarn_props_file, 20, 20 );
		
		
	}

	/**
	 * only used in development testing
	 * 
	 * @throws Exception
	 */
	@Test
	public void eval_IR_DBN_MNIST_OneWorker_TwoLabels() throws Exception {

		String yarn_props_file = "src/test/resources/run_profiles/unit_tests/dbn/mnist/app.unit_test.dbn.mnist.two_labels.properties";
		
		ModelTester.evaluateModel( yarn_props_file, 20, 20 );
		
		
	}
	
	@Test
	public void testIR_DBN_MNIST_TwoWorker_TwoLabels() throws Exception {

		// setup the two input files
		String yarn_props_file = "src/test/resources/run_profiles/unit_tests/dbn/mnist/app.unit_test.dbn.mnist.two_workers.two_labels.properties";
		
		
		// run the simulator
		
		IRUnitDriver polr_ir = new IRUnitDriver( yarn_props_file );
		polr_ir.Setup();

		polr_ir.SimulateRun();

		
		// evaluate the saved model
		// - read the saved model location from the yarn props file
		// - read the input test vectors
		
		ModelTester.evaluateModel( yarn_props_file, 20, 20 );
		
		
		
	}
	
	@Test
	public void testIR_DBN_MNIST_ThreeWorker_TwoLabels() throws Exception {

		// setup the two input files
		String yarn_props_file = "src/test/resources/run_profiles/unit_tests/dbn/mnist/app.unit_test.dbn.mnist.three_workers.two_labels.properties";
		
		
		// run the simulator
		
		IRUnitDriver polr_ir = new IRUnitDriver( yarn_props_file );
		polr_ir.Setup();

		polr_ir.SimulateRun();

		
		// evaluate the saved model
		// - read the saved model location from the yarn props file
		// - read the input test vectors
		
		ModelTester.evaluateModel( yarn_props_file, 20, 20 );
		
		
		
	}	
	
	@Test
	public void testIR_DBN_MNIST_TwoWorker_TwoLabels_UnevenSplits() throws Exception {

		// setup the two input files
		String yarn_props_file = "src/test/resources/run_profiles/unit_tests/dbn/mnist/app.unit_test.dbn.mnist.two_workers.two_labels.smallsplit.properties";
		
		// run the simulator
		IRUnitDriver polr_ir = new IRUnitDriver( yarn_props_file );
		polr_ir.Setup();

		polr_ir.SimulateRun();

		
		// evaluate the saved model
		ModelTester.evaluateModel( yarn_props_file, 20, 20 );
		
		
		
	}	
	
	/**
	 * Batch size is 10 records
	 * One split is 20 records,
	 * Other split is 11 records
	 * 
	 *  - forces to worker to have a batch that is less than the standard batch size
	 * 
	 * @throws Exception
	 */
	@Test
	public void testIR_DBN_MNIST_TwoWorker_TwoLabels_JaggedSplits() throws Exception {

		// setup the two input files
		String yarn_props_file = "src/test/resources/run_profiles/unit_tests/dbn/mnist/app.unit_test.dbn.mnist.two_workers.two_labels.jaggedsplit.properties";
		
		// run the simulator
		IRUnitDriver polr_ir = new IRUnitDriver( yarn_props_file );
		polr_ir.Setup();

		polr_ir.SimulateRun();

		
		// evaluate the saved model
		ModelTester.evaluateModel( yarn_props_file, 20, 20 );
		
		
		
	}	
	

	/**
	 * Batch size is 10 records
	 * One split is 20 records,
	 * Other split is 8 records
	 * 
	 *  - forces to worker to have a batch that is less than the standard batch size
	 * 
	 * @throws Exception
	 */
	@Test
	public void testIR_DBN_MNIST_TwoWorker_TwoLabels_JaggedSplitsSmall() throws Exception {

		// setup the two input files
		String yarn_props_file = "src/test/resources/run_profiles/unit_tests/dbn/mnist/app.unit_test.dbn.mnist.two_workers.two_labels.jaggedsplit_small.properties";
		
		// run the simulator
		IRUnitDriver polr_ir = new IRUnitDriver( yarn_props_file );
		polr_ir.Setup();

		polr_ir.SimulateRun();

		
		// evaluate the saved model
		ModelTester.evaluateModel( yarn_props_file, 20, 20 );
		
		
		
	}		
	
	
	@Test
	public void evaluateIR_DBN_TwoLabel_ScoreSavedModel() throws Exception {
/*		
		String tmpFilename = "/tmp/MNIST/dbn.mnist.twolabels.dl_model";
		String testVectors = "/tmp/mnist_filtered_conversion_test.metronome";
		int[] hiddenLayerSizes = new int[] {2,2,2};
		
		
		
		FileInputStream oFileInputStream = new FileInputStream( tmpFilename );
		
		
		DeepBeliefNetwork dbn_deserialize = new DeepBeliefNetwork(1, hiddenLayerSizes, 1, hiddenLayerSizes.length, null ); //, Matrix input, Matrix labels);
		dbn_deserialize.load(oFileInputStream);
		
		DataSet testBatch = this.setupHDFSDataset(testVectors);
		
		ModelTester.evaluateModel( testBatch.getFirst(), testBatch.getSecond(), dbn_deserialize);
		*/
		
		String yarn_props_file = "src/test/resources/run_profiles/unit_tests/dbn/mnist/app.unit_test.dbn.mnist.two_workers.two_labels.properties";
		
		ModelTester.evaluateModel( yarn_props_file, 20, 20 );
		
		
	}
	
	
}