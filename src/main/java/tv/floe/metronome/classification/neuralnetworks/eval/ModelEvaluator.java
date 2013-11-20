package tv.floe.metronome.classification.neuralnetworks.eval;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Properties;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;
import tv.floe.metronome.io.records.MetronomeRecordFactory;

public class ModelEvaluator {


	private static String model_path = "/tmp/nn.wine.model";
	private static String schema = "i:64 | o:10";
	private static String src_input_data_path = "src/test/resources/data/nist/HandWritingDigits/optdigits.tra.mne";


	//private static Properties props = null;
	//private static String app_properties_file = "";
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
	
	
	public static MultiLayerPerceptronNetwork loadModelFromDisk(String model_path) {
		
		MultiLayerPerceptronNetwork nnet = null;
		
		try {

			Path out = new Path(model_path.trim()); 
			FileSystem fs =
					  out.getFileSystem(defaultConf); 
			/*
			FSDataOutputStream fos;

			fos = fs.create(out);
			  //LOG.info("Writing master results to " + out.toString());
			  mnode.complete(fos);
			  
			  fos.flush(); 
			  fos.close();
*/
			//BufferedWriter bw = new BufferedWriter(new FileWriter(output_path));
			//master.complete( bw );
			
			//bw.close();
			System.out.println("Loading Model: " + out.toString().replaceAll("file:", ""));
			FileInputStream fis = new FileInputStream( out.toString().replaceAll("file:", "") );
		    //ObjectInputStream ois = new ObjectInputStream(fis);
		    
		    //oos.writeObject( master. );
		    //master.complete(oos);
		    
		    //nnet = (MultiLayerPerceptronNetwork) ois.readObject();
		    byte[] b = new byte[1024 * 200];
		    int bytes_read = fis.read(b);
		    
		    //System.out.println("bytes read: " + bytes_read);
		    
		    nnet = (MultiLayerPerceptronNetwork) NeuralNetwork.Deserialize(b);
		    
		    //oos.flush();
		    //ois.close();

			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
		
		return nnet;
		
	}
	
	public static void scoreNeuralNetworkModel(String modelFileLocation, String schema, String records_file) throws Exception {
		
		// load model
		MultiLayerPerceptronNetwork mlp = loadModelFromDisk(modelFileLocation);
		
		//Utils.PrintNeuralNetwork(  mlp );
		
		//System.out.println("Layers: " + mlp.getLayersCount());
		
		MetronomeRecordFactory rec_factory = new MetronomeRecordFactory(schema);
	
			
			String recs = records_file;
			Vector v_in_0 = new RandomAccessSparseVector(rec_factory.getInputVectorSize());
			Vector v_out_0 = new RandomAccessSparseVector(rec_factory.getOutputVectorSize());
			
			
			
			int total_recs = 0;
			int correct = 0;
			
			BufferedReader br = new BufferedReader(new FileReader(recs));
			String line;
			while ((line = br.readLine()) != null) {
				
				

				rec_factory.vectorizeLine( line, v_in_0, v_out_0 );
				
				if (isCorrect(mlp, v_in_0, v_out_0) ) {
					correct++;
				}
				total_recs++;
				
				//System.out.println("rec > " + line);
				//System.out.println("out > " + v_out_0.toString());
				
			}
			br.close();		
			
			System.out.println("Total: " + total_recs);
			System.out.println("Correct: " + correct);
		
			double percent = (double)correct / (double)total_recs;
		
			System.out.println("Percent: " + percent);
			
	}
	
	public static int getLargestIndex(Vector v) {
		
		double largest_val = 0;
		int largest_index = 0;
		
		for (int x = 0; x < v.size(); x++) {
			
			if ( v.get(x) > largest_val ) {
				largest_val = v.get(x);
				largest_index = x;
			}
			
		}
		
		return largest_index;
	}
	
	public static boolean isCorrect(NeuralNetwork mlp, Vector input_vec, Vector answer_vec) throws Exception {
		
		mlp.setInputVector( input_vec );
		mlp.calculate();
        Vector networkOutput = mlp.getOutputVector();

        
		
        Vector v_answer_selected = new RandomAccessSparseVector(answer_vec.size());
        
        v_answer_selected.set(getLargestIndex(networkOutput), 1.0);
        
        //System.out.println( "> out: " + v_answer_selected.toString() );
        
        if ( answer_vec.equals(v_answer_selected)) {
        	return true;
        }
        
        
		return false;
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

		schema = props.getProperty("tv.floe.metronome.neuralnetwork.conf.InputRecordSchema");
		if (null == schema) {
			throw new Exception("Can't find the input record schema in the properites file!");
		}

		src_input_data_path = props.getProperty("tv.floe.metronome.neuralnetwork.conf.evaluate.dataset.path");
		if (null == src_input_data_path) {
			throw new Exception("Can't find the eval/test recordset in the properites file!");
		}
/*		
		System.out.println("Conf ------------");
		System.out.println("Model: " + model_path);
		System.out.println("Schema: " + schema);
		System.out.println("Test Dataset: " + src_input_data_path);
	*/	
		
	}
	
	public static void eval(String propertiesFile) throws Exception {
		
		parsePropertiesFile(propertiesFile);
		
		scoreNeuralNetworkModel( model_path, schema, src_input_data_path );
		
	}
}
