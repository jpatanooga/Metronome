package tv.floe.metronome.deeplearning.datasets.fetchers;

import java.io.Serializable;

import tv.floe.metronome.deeplearning.datasets.DataSet;


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
