package tv.floe.metronome.deeplearning.datasets.iterator.impl;

import java.io.IOException;

import com.cloudera.iterativereduce.io.TextRecordParser;

import tv.floe.metronome.deeplearning.datasets.fetchers.MnistHDFSDataFetcher;
import tv.floe.metronome.deeplearning.datasets.iterator.BaseDatasetIterator;

public class MnistHDFSDataSetIterator extends BaseDatasetIterator {

	public MnistHDFSDataSetIterator( int batch,int numExamples, TextRecordParser hdfsLineParser ) throws IOException {
		
		super( batch, numExamples, new MnistHDFSDataFetcher( hdfsLineParser ) );
		
		
	}
	
	@Override
	public boolean hasNext() {
		return fetcher.hasMore();
	}
	

}
