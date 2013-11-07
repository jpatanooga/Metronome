package tv.floe.metronome.classification.neuralnetworks.iterativereduce.bcw;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;
import tv.floe.metronome.io.records.MetronomeRecordFactory;

public class BreastCancerWisconsin_DatasetUtils {


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
	/**
	 * 10 columns
	 * 9 input coluns, 2 output columns
	 * 
	 * @param line
	 * @return
	 * @throws Exception 
	 */
	public static String formatIrisColumnToMetronomeRecord(String line, int normalize_base) throws Exception {
		/*
		String[] parts = line.split(",");

		String inputs = "";
		inputs += "0:" + parts[0].trim() + " ";
		inputs += "1:" + parts[1].trim() + " ";
		inputs += "2:" + parts[2].trim() + " ";
		inputs += "3:" + parts[3].trim() + " ";
		
		String outputs = "";
		outputs += "0:" + parts[4].trim() + " ";
		outputs += "1:" + parts[5].trim() + " ";
		outputs += "2:" + parts[6].trim() + "";
		
		return inputs + "| " + outputs + "\n";
		*/
		

		String[] parts = line.split(",");

		String inputs = "";
		
		int max = 0;
		
		//System.out.println("len: " + parts.length);
		//return "";
		
		if (parts.length != 11) {
			System.out.println("Invalid Line!!!!!");
			throw new Exception("Invalid line format for Breast Cancer Wisconsin dataset conversion!");
		}
		

		for ( int x = 1; x < 10; x++ ) {
			//inputs += x + ":" + parts[x].trim() + " ";
			
			int tmp = 0; 
			
			if ("?".equals(parts[x])) {
				tmp = 1;
			} else {
				tmp = Integer.parseInt(parts[x]);
			}
			
/*			if (tmp_max > max) {
				max = tmp_max;
			}
			*/
			double normalized_val = (double)tmp / (double)normalize_base;
			if (normalized_val > 0.0) {
				inputs += (x-1) + ":" + normalized_val + " ";
			}
		}
		
		//System.out.println("max: " + max);
		
		String class_id = parts[10]; // last one
		int cls_id = Integer.parseInt(class_id);

		String outputs = "";

		if (cls_id == 4) {
			outputs += "0:1.0";
			
		} else if (cls_id == 2) {
			outputs += "0:0.0";
		} else {
			throw new Exception("Invalid class id!");
		}
//		outputs += "1:" + parts[5].trim() + " ";
//		outputs += "2:" + parts[6].trim() + "";
		
		return inputs + "| " + outputs + "\n";		
		
		
	}
	/**
	 * Example:
	 * 0.64556962, 0.795454545, 0.202898551, 0.08, 1, 0, 0
	 * 
	 * 7 columns in dataset
	 * 
	 * @param filename
	 * @throws Exception 
	 */
	public static void convertBCWNormalizedToMetronome(String filename, String file_out) throws Exception {
		
		// src/test/resources/data/iris/iris_data_normalised.txt
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(file_out));
		
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line;
		while ((line = br.readLine()) != null) {
		   // process the line.
			
			String formatted_line = formatIrisColumnToMetronomeRecord(line, 10);
			
		//	System.out.println("old > " + line);
		//	System.out.println("new > " + formatted_line);
			bw.write(formatted_line);
			
		}
		br.close();		
		bw.close();
		
		
		
	}
	

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		convertBCWNormalizedToMetronome("src/test/resources/data/uci/breast_cancer_wisconsin/breast-cancer-wisconsin.data.txt", "src/test/resources/data/uci/breast_cancer_wisconsin/bcw.mne");
		
	}

}
