package tv.floe.metronome.clustering.kmeans;

import static org.junit.Assert.*;

import org.junit.Test;

import tv.floe.metronome.clustering.kmeans.Mean;
import tv.floe.metronome.clustering.kmeans.Point;

public class MeanTest {

	@Test
	public void test() {
		Point zero = new Point(0,0);
		Point one = new Point(1,1);
		
		Point half = new Point(0.5, 0.5);
		Point quarter = new Point(0.25, 0.25);
		
		Mean averager = new Mean();
		averager.add(zero);
		averager.add(one);
		
		Point averageA = averager.toPoint();
		assertEquals(half, averageA);
		
		averager.reset();
		averager.add(zero);
		averager.add(zero);
		averager.add(zero);
		averager.add(one);
		
		Point averageB = averager.toPoint();
		assertEquals(quarter, averageB);
	}
	
	
	@Test
	public void testMerge() {
		Point zero = new Point(0,0);
		Point one = new Point(1,1);
		
		Point half = new Point(0.5, 0.5);
		
		Mean averagerA = new Mean();
		averagerA.add(zero);
		averagerA.add(zero);
		averagerA.add(one);
		
		Mean averagerB = new Mean();
		averagerB.add(zero);
		averagerB.add(one);
		averagerB.add(one);

		averagerA.merge(averagerB);
		
		Point average = averagerA.toPoint();
		assertEquals(half, average);
		
	}
	
	@Test(expected=IllegalStateException.class)
	public void testToPoint() {
		Mean mean = new Mean();
		mean.toPoint();
	}

}
