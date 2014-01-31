package tv.floe.metronome.deeplearning.datasets.iterator.impl;

import java.io.IOException;

import tv.floe.metronome.deeplearning.datasets.fetchers.MnistDataFetcher;
import tv.floe.metronome.deeplearning.datasets.iterator.BaseDatasetIterator;
import tv.floe.metronome.deeplearning.datasets.iterator.DataSetFetcher;


public class MnistDataSetIterator extends BaseDatasetIterator {

	public MnistDataSetIterator( int batch,int numExamples ) throws IOException {
		
		super( batch, numExamples, (DataSetFetcher) new MnistDataFetcher() );
		
		
	}


}
