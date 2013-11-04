package tv.floe.metronome.io.records;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class DatasetConverter {
	
	/**
	 * 7 columns
	 * 4 input coluns, 3 output columns
	 * 
	 * @param line
	 * @return
	 */
	public static String formatIrisColumnToMetronomeRecord(String line) {
		
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
			
			String formatted_line = formatIrisColumnToMetronomeRecord(line);
			
		//	System.out.println("old > " + line);
		//	System.out.println("new > " + formatted_line);
			bw.write(formatted_line);
			
		}
		br.close();		
		bw.close();
		
		
		
	}
	

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub

		convertIrisNormalizedToMetronome("src/test/resources/data/iris/iris_data_normalised.txt", "src/test/resources/data/iris/iris_data_normalised.mne");
		
	}

}
