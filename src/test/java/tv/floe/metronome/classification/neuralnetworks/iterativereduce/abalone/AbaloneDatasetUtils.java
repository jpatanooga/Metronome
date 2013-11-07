package tv.floe.metronome.classification.neuralnetworks.iterativereduce.abalone;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapred.JobConf;

public class AbaloneDatasetUtils {



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
	 * 8 columns
	 * 29 classes
	 * 
	 * @param line
	 * @return
	 * @throws Exception 
	 */
	public static String formatToMetronomeRecord(String line) throws Exception {

		
		

		//String line2 = line.replaceAll("  ", ",").replaceAll(" ", ",");

		//System.out.println(line2);
		
		String[] parts = line.split(",");

		String inputs = "";
		
		int max = 0;
		
		//System.out.println("len: " + parts.length);
		//return "";
		
		if (parts.length != 9) {
			System.out.println("Invalid Line: " + line + ", len: " + parts.length);
			throw new Exception("Invalid line format for UCI Lenses dataset conversion!");
		}
		

		for ( int x = 0; x < 8; x++ ) {
			//inputs += x + ":" + parts[x].trim() + " ";
			
			double tmp = 0; 
			
			if (x == 0) {
			
				if ("m".equals(parts[x].trim().toLowerCase())) {
					
					tmp = 0.0;
					
				} else if ("f".equals(parts[x].trim().toLowerCase())) {
					
					tmp = 0.5;
					
				} else {
					
					tmp = 1.0;
					
				}
				
			} else if ("?".equals(parts[x])) {
				tmp = 1;
			} else {
				tmp = Double.parseDouble(parts[x]);
			}
			
/*			if (tmp_max > max) {
				max = tmp_max;
			}
			*/
			double normalized_val = tmp;
			if (normalized_val > 0.0) {
				inputs += (x) + ":" + normalized_val + " ";
			} else {
				//System.out.println("omit: " + x + " == " + tmp);
			}
		}
		
		//System.out.println("max: " + max);
		
		String class_id = parts[8]; // last one
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
		
		int recs = 0;
		
		// src/test/resources/data/iris/iris_data_normalised.txt
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(file_out));
		
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line;
		while ((line = br.readLine()) != null) {
		   // process the line.
			
			String formatted_line = formatToMetronomeRecord(line);
			
		//	System.out.println("old > " + line);
		//	System.out.println("new > " + formatted_line);
			bw.write(formatted_line);
			
			recs++;
			
		}
		br.close();		
		bw.close();
		
		System.out.println("> " + recs + " converted");
		
		
	}
	

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		convertLensesNormalizedToMetronome("src/test/resources/data/uci/abalone/original/abalone.data.txt", "src/test/resources/data/uci/abalone/oneworker/abalone.mne");
		
	}
}
