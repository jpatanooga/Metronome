package tv.floe.metronome.deeplearning.datasets.iterator;

import java.io.Serializable;
import java.util.Iterator;

import tv.floe.metronome.deeplearning.datasets.DataSet;

/**
 * Originally from dl4j
 * - https://github.com/agibsonccc/java-deeplearning
 * 
 * @author Adam Gibson
 *
 */
public interface DataSetIterator extends Iterator<DataSet>,Serializable {

	
	int totalExamples();
	
	int inputColumns();
	
	int totalOutcomes();
	
	void reset();
	
	int batch();
	
	int cursor();
	
	int numExamples();
	
}
