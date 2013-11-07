package tv.floe.metronome.classification.neuralnetworks.iterativereduce.nist.handwriting;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class NIST_HandWritingDatasetUtils {

	
	/**
	 * 64 input+1 class attribute
	 * 
	 * 
	 * @param line
	 * @return
	 */
	public static String formatNISTHandwritingColumnToMetronomeRecord(String line, int normalize_base) {
		
		String[] parts = line.split(",");

		String inputs = "";
		
		int max = 0;
		
		//System.out.println("len: " + parts.length);
		//return "";
		
		if (parts.length != 65) {
			System.out.println("Invalid Line!!!!!");
		}
		

		for ( int x = 0; x < 64; x++ ) {
			//inputs += x + ":" + parts[x].trim() + " ";
			int tmp = Integer.parseInt(parts[x]);
/*			if (tmp_max > max) {
				max = tmp_max;
			}
			*/
			double normalized_val = (double)tmp / (double)normalize_base;
			if (normalized_val > 0.0) {
				inputs += x + ":" + normalized_val + " ";
			}
		}
		
		//System.out.println("max: " + max);
		
		String class_id = parts[64]; // last one
		int cls_id = Integer.parseInt(class_id);
		
		
		String outputs = "";
		outputs += cls_id + ":1.0";
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
	 * @throws IOException
	 */
	public static void convertIrisNormalizedToMetronome(String filename, String file_out) throws IOException {
		
		// src/test/resources/data/iris/iris_data_normalised.txt
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(file_out));
		
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line;
		while ((line = br.readLine()) != null) {
		   // process the line.
			
			String formatted_line = formatNISTHandwritingColumnToMetronomeRecord(line, 16);
			
			System.out.println("old > " + line);
			System.out.println("new > " + formatted_line);
			bw.write(formatted_line);
			
		}
		br.close();		
		bw.close();
		
		System.out.println("done...");
		
	}	
	
	
	
	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub

		convertIrisNormalizedToMetronome("src/test/resources/data/NIST/HandwritingDigits/optdigits.tra", "src/test/resources/data/NIST/HandwritingDigits/optdigits.tra.mne");
		
	}

}
