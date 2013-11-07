package tv.floe.metronome.classification.neuralnetworks.iterativereduce.uci.dermatology;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapred.JobConf;

public class DermatologyDatasetUtils {


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
	 * 34 columns
	 * class: column 0
	 * 6 classes
	 * 
	 * @param line
	 * @return
	 * @throws Exception 
	 */
	public static String formatToMetronomeRecord(String line, double[] normalize_bases) throws Exception {

		
		

		//String line2 = line.replaceAll("  ", ",").replaceAll(" ", ",");

		//System.out.println(line2);
		
		String[] parts = line.split(",");

		String inputs = "";
		
		int max = 0;
		int cols = 35;
		
		//System.out.println("len: " + parts.length);
		//return "";
		
		if (parts.length != cols) {
			System.out.println("Invalid Line: " + line + ", len: " + parts.length);
			throw new Exception("Invalid line format for UCI Wine dataset conversion!");
		}
		

		for ( int x = 0; x < cols; x++ ) {
			//inputs += x + ":" + parts[x].trim() + " ";
			
			double tmp = 0; 
			
			if (x == 0) {
			
				
			} else if ("?".equals(parts[x])) {
				tmp = 0.5 * normalize_bases[x - 1];
			} else {
				tmp = Double.parseDouble(parts[x]);

				double normalized_val = tmp / normalize_bases[x - 1];
				if (normalized_val > 0.0) {
					inputs += (x-1) + ":" + normalized_val + " ";
				} else {
					//System.out.println("omit: " + x + " == " + tmp);
				}

			}
			
		}
		
		//System.out.println("max: " + max);
		
		String class_id = parts[34]; // last one
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
	
	
	public static double[] scanFileForNormalizationColumnBases(String filename) throws IOException {
		
		int class_index = 34;
		int columns = 35;
		int norm_bases = columns - 1;
		
		double[] bases = new double[norm_bases];
		
		// init
		for ( int x = 0; x < norm_bases; x++ ) {
			
			bases[x] = 0;
			
		}
		
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line;
		while ((line = br.readLine()) != null) {
		   // process the line.
			
			// split to floats
			String[] parts = line.split(",");
			
			for ( int x = 0; x < columns - 1; x++ ) {
				
				if ("?".equals(parts[x])) {
					
				} else { 
					double val = Double.parseDouble(parts[x]);
					
					if (val > bases[x]) { 
						bases[ x ] = val;
					}
					
				}
				
				
			}
			
			
			//String formatted_line = formatToMetronomeRecord(line);
			
		//	System.out.println("old > " + line);
		//	System.out.println("new > " + formatted_line);
			//bw.write(formatted_line);
			
			//recs++;
			
		}
		br.close();			
		
		return bases;
		
	}
	
	/**
	 * Example:
	 * 
	 * @param filename
	 * @throws Exception 
	 */
	public static void convertToNormalizedMetronome(String filename, String file_out) throws Exception {
		
		int recs = 0;
		
		// src/test/resources/data/iris/iris_data_normalised.txt
		
		double bases[] = scanFileForNormalizationColumnBases(filename);
		
		System.out.println("Bases: " + bases.length);
		for ( int x = 0; x < bases.length; x++ ) {
			System.out.println("> " + x + ": " + bases[x]);
		}
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(file_out));
		
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line;
		while ((line = br.readLine()) != null) {
		   // process the line.
			
			String formatted_line = formatToMetronomeRecord(line, bases);
			
			//System.out.println("old > " + line);
			//System.out.println("new > " + formatted_line);
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

		convertToNormalizedMetronome("src/test/resources/data/uci/dermatology/original/dermatology.data.txt", "src/test/resources/data/uci/dermatology/oneworker/dermatology.mne");
		
	}
}
