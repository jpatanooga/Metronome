package tv.floe.metronome.deeplearning.dbn.model.evaluation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Properties;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.mahout.math.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.cloudera.iterativereduce.io.TextRecordParser;

import tv.floe.metronome.berkley.Pair;
import tv.floe.metronome.deeplearning.datasets.DataSet;
import tv.floe.metronome.deeplearning.datasets.iterator.BaseDatasetIterator;
import tv.floe.metronome.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import tv.floe.metronome.deeplearning.datasets.iterator.impl.MnistHDFSDataSetIterator;
import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;
import tv.floe.metronome.deeplearning.neuralnetwork.core.BaseMultiLayerNeuralNetworkVectorized;
import tv.floe.metronome.eval.Evaluation;

/**
 * Model tester build specifically for Deep Belief Networks
 * 
 * @author josh
 *
 */
public class ModelTester {
	

	private static Logger log = LoggerFactory.getLogger(ModelTester.class);
	
	public static String model_path = "";
	public static String test_input_data_path = "";
	
	


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
	
	
	private static InputSplit[] generateDebugSplits(Path input_path, JobConf job) {

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
	
	
	
	public static void evaluateModel( BaseDatasetIterator iterator, BaseMultiLayerNeuralNetworkVectorized model ) throws IOException {
		

		Evaluation eval = new Evaluation();
		//BaseMultiLayerNeuralNetworkVectorized load = BaseMultiLayerNeuralNetworkVectorized.loadFromFile(new FileInputStream(new File(modelLocation)));
		
		while (iterator.hasNext()) {
			
			DataSet inputs = iterator.next();

			Matrix in = inputs.getFirst();
			Matrix outcomes = inputs.getSecond();
			Matrix predicted = model.predict(in);
			eval.eval( outcomes, predicted );
			
		}
		
		
		
		log.warn( "evaluateModel" );
		log.info( eval.stats() );		
		
		//writeReportToDisk( eval, pathForReport );
		
	}	
	
	
	
	public static void evaluateModel( Matrix inputs, Matrix labels, BaseMultiLayerNeuralNetworkVectorized model ) throws IOException {
		

		Evaluation eval = new Evaluation();
		//BaseMultiLayerNeuralNetworkVectorized load = BaseMultiLayerNeuralNetworkVectorized.loadFromFile(new FileInputStream(new File(modelLocation)));
		
//		while (iterator.hasNext()) {
			
//			DataSet inputs = iterator.next();

//			Matrix in = inputs.getFirst();
//			Matrix outcomes = inputs.getSecond();

		Matrix predicted = model.predict(inputs);
		
		eval.eval( labels, predicted );
			
//		}
		
		
		
		log.warn( "evaluateModel" );
		log.info( eval.stats() );		
		
		//writeReportToDisk( eval, pathForReport );
		
	}		
	
	public static void evaluateSavedModel( BaseDatasetIterator iterator, String modelLocation, String pathForReport ) throws IOException {
		

		Evaluation eval = new Evaluation();
		BaseMultiLayerNeuralNetworkVectorized load = BaseMultiLayerNeuralNetworkVectorized.loadFromFile(new FileInputStream(new File(modelLocation)));
		
		while (iterator.hasNext()) {
			
			DataSet inputs = iterator.next();

			Matrix in = inputs.getFirst();
			Matrix outcomes = inputs.getSecond();
			Matrix predicted = load.predict(in);
			eval.eval( outcomes, predicted );
			
		}
		
		
		
		
		log.info( eval.stats() );		
		
		writeReportToDisk( eval, pathForReport );
		
	}
	
	public static void evaluateModel(String propsFilepath, Configuration hadoopConf, int batchSize ) throws Exception {

		
		parsePropertiesFile(propsFilepath);
		
		
	
	
	
		// setup splits ala HDFS style -------------------
		
	    JobConf job = new JobConf(defaultConf);
	    if (null != hadoopConf ) {
	    	job = new JobConf( hadoopConf );
	    } else {
	    	hadoopConf = defaultConf;
	    }
	    
	    Path workDir = new Path( test_input_data_path );
		
		
	    InputSplit[] splits = generateDebugSplits(workDir, job);
	    
	    System.out.println( "> splits: " + splits[0].toString() );

	    
	    TextRecordParser txt_reader = new TextRecordParser();

	    // drop the fully qualified host
	    //String minus_host = splits[0].toString().split( "" )
	    
	    long len = 0;
	    String path = "";
	    
	    // YES, more clever regex here would be clever.
	    // I'm on a plane right now and lazy. sue me.

	    if ( splits[0].toString().startsWith( "file:///" ) ) {
	    	
	    	path = splits[0].toString().replaceFirst("file:///", "/").split(":")[0];
	    	
	    	len = splits[0].getLength();
	    	
	    } else if ( splits[0].toString().startsWith( "file://" ) ) {
		    	
		    	path = splits[0].toString().replaceFirst("file://", "/").split(":")[0];
		    	
		    	len = splits[0].getLength();
	    
	    } else if ( splits[0].toString().startsWith( "file:/" ) ) {
	    	
	    	path = splits[0].toString().replaceFirst("file:/", "/").split(":")[0];
	    	
	    	len = splits[0].getLength();
	    
	    } else if ( splits[0].toString().substring(1).equals( "/" ) ) {
	    
	    	// we have something like:
	    	// /user/cloudera/metronome/dbn/mnist/mnist_filtered_conversion_test.metronome:0+20908
	    	
	    	len = Integer.parseInt(splits[0].toString().split(":")[2]
	    			.split("\\+")[1]);
	    	
	    	System.out.println("Split Length: " + len + ", versus method: " + splits[0].getLength() );
	    	
	    	path = splits[0].toString().split(":")[0];
	    	
	    } else if (splits[0].toString().substring(8).equals( "hdfs:///" ) ) {
	    	
	    	// we have something like:
	    	// hdfs:///user/cloudera/metronome/dbn/mnist/mnist_filtered_conversion_test.metronome:0+20908
	    	
	    	
	    	len = Integer.parseInt(splits[0].toString().split(":")[2]
	    			.split("\\+")[1]);

	    	System.out.println( "number of slices from ':' " + splits[0].toString().split(":").length );
	    	
	    	System.out.println("Split Length: " + len + ", versus method: " + splits[0].getLength() );
	    	
	    	path = splits[0].toString().split(":")[1];
	    	
	    } else {
	    	
	    	// we have something like:
	    	// hdfs://localhost.localdomain:8020/user/cloudera/metronome/dbn/mnist/mnist_filtered_conversion_test.metronome:0+20908
	    	
	    	len = Integer.parseInt(splits[0].toString().split(":")[3]
	    			.split("\\+")[1]);

	    	System.out.println( "number of slices from ':' " + splits[0].toString().split(":").length );
	    	
	    	System.out.println("Split Length: " + len + ", versus method: " + splits[0].getLength() );
	    	
	    	String[] parts = splits[0].toString().split(":");
	    	
	    	path = parts[0] + ":" + parts[1] + ":" + parts[2];
	    	
	    	
	    }
	    
	    System.out.println("raw path: " + path);
	    

	  //  txt_reader.setFile(splits[0].toString().split(":")[1], 0, len);		
	    txt_reader.setFile( path, 0, len);
		
					

	    
		MnistHDFSDataSetIterator hdfs_fetcher = new MnistHDFSDataSetIterator( batchSize, 1, txt_reader );
		DataSet hdfs_recordBatch = hdfs_fetcher.next();
		
		Matrix hdfs_input = hdfs_recordBatch.getFirst();
		Matrix hdfs_labels = hdfs_recordBatch.getSecond();			
		
		//ModelTester.evaluateModel( recordBatch.getFirst(), recordBatch.getSecond(), dbn);
			
		
		int[] hiddenLayerSizes = new int[] {2,2,2};
		
		if (model_path.startsWith("file://")) {
			model_path = model_path.replaceFirst("file://", "");
		}
		
		System.out.println("Evaluating DBN Model Saved at: " + model_path );
		
		Path modelInputPath = new Path( model_path );
		
	//	System.out.println("Evaluating DBN Model Saved at: " + modelInputPath.getName() );
	//	System.out.println("Evaluating DBN Model Saved at: " + modelInputPath.toString() );
	//	System.out.println("Evaluating DBN Model Saved at: " + modelInputPath.toUri().toString() );
		
//		FileInputStream oFileInputStream = new FileInputStream( modelInputPath.toUri().toString().replaceAll( "hdfs:///" , "hdfs://localhost.localdomain:8020/" ) );
		
		Path model_new_path = new Path( model_path );
	    FileSystem fs = model_new_path.getFileSystem( hadoopConf );
	    FSDataInputStream inputStream = fs.open( model_new_path);		
		
		
		DeepBeliefNetwork dbn_deserialize = new DeepBeliefNetwork(1, hiddenLayerSizes, 1, hiddenLayerSizes.length, null ); //, Matrix input, Matrix labels);
		//dbn_deserialize.load(oFileInputStream);
		dbn_deserialize.load( inputStream );
		
		evaluateModel( hdfs_input, hdfs_labels, dbn_deserialize );
			
			
		
	}
	

	public static void parsePropertiesFile(String app_properties_file) throws Exception {
		
		Properties props = new Properties();
		// Configuration conf = getConf();

		try {
			FileInputStream fis = new FileInputStream(app_properties_file);
			props.load(fis);
			fis.close();
		} catch (FileNotFoundException ex) {
			// throw ex; // TODO: be nice
			System.out.println(ex);
		} catch (IOException ex) {
			// throw ex; // TODO: be nice
			System.out.println(ex);
		}	
		
		model_path = props.getProperty("app.output.path");
		if (null == model_path) {
			throw new Exception("Can't find the model output path in the properites file!");
		}

/*		schema = props.getProperty("tv.floe.metronome.neuralnetwork.conf.InputRecordSchema");
		if (null == schema) {
			throw new Exception("Can't find the input record schema in the properites file!");
		}
*/
		test_input_data_path = props.getProperty("tv.floe.metronome.evaluate.dataset.path");
		if (null == test_input_data_path) {
			throw new Exception("Can't find the eval/test recordset in the properites file!");
		}

		
	}	
		
	
	public static void writeReportToDisk( Evaluation eval, String fileLocation ) throws IOException {
		
		// open files somewhere
		
		File yourFile = new File(fileLocation);
		if(!yourFile.exists()) {
		    yourFile.createNewFile();
		} 
		FileOutputStream oFile = new FileOutputStream(fileLocation, false); 
		
		oFile.write(eval.stats().getBytes() );
		
		oFile.close();
		
		
	}
	
	
	/**
	 * @param args
	 * @throws IOException 
	 */
/*	public static void main(String[] args) throws IOException {
		
		MnistDataSetIterator iter = new MnistDataSetIterator(10, 60000);
		
		Evaluation eval = new Evaluation();
		BaseMultiLayerNeuralNetworkVectorized load = BaseMultiLayerNeuralNetworkVectorized.loadFromFile(new FileInputStream(new File(args[0])));
		
		while (iter.hasNext()) {
			
			DataSet inputs = iter.next();

			Matrix in = inputs.getFirst();
			Matrix outcomes = inputs.getSecond();
			Matrix predicted = load.predict(in);
			eval.eval( outcomes, predicted );
			
		}
		
		
		
		
		log.info( eval.stats() );
	}	
*/	

}
