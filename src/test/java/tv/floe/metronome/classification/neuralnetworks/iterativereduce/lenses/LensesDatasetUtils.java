package tv.floe.metronome.classification.neuralnetworks.iterativereduce.lenses;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapred.JobConf;

public class LensesDatasetUtils {



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
	public static String formatToMetronomeRecord(String line, int normalize_base) throws Exception {
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
		

		String line2 = line.replaceAll("  ", ",").replaceAll(" ", ",");

		//System.out.println(line2);
		
		String[] parts = line2.split(",");

		String inputs = "";
		
		int max = 0;
		
		//System.out.println("len: " + parts.length);
		//return "";
		
		if (parts.length != 6) {
			System.out.println("Invalid Line: " + line2 + ", len: " + parts.length);
			throw new Exception("Invalid line format for UCI Lenses dataset conversion!");
		}
		

		for ( int x = 1; x < 5; x++ ) {
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
			} else {
				System.out.println("omit: " + x + " == " + tmp);
			}
		}
		
		//System.out.println("max: " + max);
		
		String class_id = parts[5]; // last one
		int cls_id = Integer.parseInt(class_id);

		String outputs = "";

		//if (cls_id == 4) {
			outputs += (cls_id - 1) + ":1.0";
			
/*		} else if (cls_id == 2) {
			outputs += "0:0.0";
		} else {
			throw new Exception("Invalid class id!");
		}
		*/
//		outputs += "1:" + parts[5].trim() + " ";
//		outputs += "2:" + parts[6].trim() + "";
		
		return inputs + "| " + outputs + "\n";		
		
		
	}
	/**
	 * Example:
	 * 
	 * @param filename
	 * @throws Exception 
	 */
	public static void convertLensesNormalizedToMetronome(String filename, String file_out) throws Exception {
		
		// src/test/resources/data/iris/iris_data_normalised.txt
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(file_out));
		
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line;
		while ((line = br.readLine()) != null) {
		   // process the line.
			
			String formatted_line = formatToMetronomeRecord(line, 3);
			
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

		convertLensesNormalizedToMetronome("src/test/resources/data/uci/lenses/original/lenses.data.txt", "src/test/resources/data/uci/lenses/oneworker/lenses.mne");
		
	}
}
