package tv.floe.metronome.deeplearning.datasets.iterator.impl;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeNotNull;

import org.junit.Before;
import org.junit.Test;

import tv.floe.metronome.deeplearning.datasets.iterator.DataSetIterator;


public abstract class BaseDataSetIteratorTest {

	protected DataSetIterator iter;
	
	@Before
	public void setup() {
		iter = getIter();
		assumeNotNull(iter);
	}
	
	
	protected void testInit(DataSetIterator fetcher,int inputColumnsExpected,int totalOutcomesExpected,int batchExpected) {
		assertEquals(inputColumnsExpected,fetcher.inputColumns());
		assertEquals(totalOutcomesExpected,fetcher.totalOutcomes());
		assertEquals(true,fetcher.hasNext());
		assertEquals(true,fetcher.next() != null && fetcher.hasNext());
		assertEquals(batchExpected,fetcher.batch());
	}


	
	
	protected void testReset(DataSetIterator toTest,int expectedCursorPosition) {
		toTest.reset();
		assertEquals(true,toTest.hasNext());
		assertEquals(expectedCursorPosition,toTest.cursor());
	}
	
	protected void testNumIters(DataSetIterator iter,int numIters,boolean shouldHaveNext) {
		assertEquals(true,iter.hasNext());
		for(int i = 0; i < numIters; i++)
			 iter.next();
		assertEquals(shouldHaveNext,iter.hasNext());
	}
	
	protected void testCursorPosition(DataSetIterator iter,int numIters,int expectedPosition) {
		assertEquals(true,iter.hasNext());
		for(int i = 0; i < numIters; i++)
			 iter.next();
		assertEquals(expectedPosition,iter.cursor());
	}
	
	
	
	
	
	public abstract DataSetIterator getIter();
}