package tv.floe.metronome.classification.neuralnetworks.iterativereduce.iris;

import static org.junit.Assert.assertEquals;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;
import tv.floe.metronome.classification.neuralnetworks.utils.Utils;
import tv.floe.metronome.io.records.MetronomeRecordFactory;

public class IrisDatasetUtils {


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
	
	
	public static MultiLayerPerceptronNetwork loadModelFromDisk() {
		
		MultiLayerPerceptronNetwork nnet = null;
		
		try {

			Path out = new Path("/tmp/nn.model"); 
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
			
			FileInputStream fis = new FileInputStream( out.toString() );
		    //ObjectInputStream ois = new ObjectInputStream(fis);
		    
		    //oos.writeObject( master. );
		    //master.complete(oos);
		    
		    //nnet = (MultiLayerPerceptronNetwork) ois.readObject();
		    byte[] b = new byte[1024 * 16];
		    int bytes_read = fis.read(b);
		    
		    System.out.println("bytes read: " + bytes_read);
		    
		    nnet = (MultiLayerPerceptronNetwork) NeuralNetwork.Deserialize(b);
		    
		    //oos.flush();
		    //ois.close();

			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
		
		return nnet;
		
	}
	
	public static void scoreIrisNeuralNetworkModel(String modelFileLocation) throws Exception {
		
		// load model
		MultiLayerPerceptronNetwork mlp = loadModelFromDisk();
		
		//Utils.PrintNeuralNetwork(  mlp );
		
		//System.out.println("Layers: " + mlp.getLayersCount());
		
		MetronomeRecordFactory rec_factory = new MetronomeRecordFactory("i:4 | o:3");
	
			
			String recs = "src/test/resources/data/iris/iris_data_normalised.mne";
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
        
        System.out.println( "> out: " + v_answer_selected.toString() );
        
        if ( answer_vec.equals(v_answer_selected)) {
        	return true;
        }
        
        
		return false;
	}
	
	public static void main(String[] args) throws Exception {
		
		scoreIrisNeuralNetworkModel("");
		
		//Vector
		
	}
	

}
