package tv.floe.metronome.deeplearning.datasets.iterator;

import java.io.Serializable;

import tv.floe.metronome.deeplearning.datasets.DataSet;

/**
 * Originally from dl4j
 * - https://github.com/agibsonccc/java-deeplearning
 * 
 * @author Adam Gibson
 *
 */
public interface DataSetFetcher extends Serializable {

	
	boolean hasMore();
	
	DataSet next();
	
	void fetch(int numExamples);
	
	int totalOutcomes();
	
	int inputColumns();
	
	int totalExamples();
	
	void reset();
	
	int cursor();
}
