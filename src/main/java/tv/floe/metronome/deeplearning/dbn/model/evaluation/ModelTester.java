package tv.floe.metronome.deeplearning.dbn.model.evaluation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import org.apache.mahout.math.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tv.floe.metronome.berkley.Pair;
import tv.floe.metronome.deeplearning.datasets.DataSet;
import tv.floe.metronome.deeplearning.datasets.iterator.BaseDatasetIterator;
import tv.floe.metronome.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import tv.floe.metronome.deeplearning.neuralnetwork.core.BaseMultiLayerNeuralNetworkVectorized;
import tv.floe.metronome.eval.Evaluation;


public class ModelTester {
	

	private static Logger log = LoggerFactory.getLogger(ModelTester.class);
	
	
	
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
