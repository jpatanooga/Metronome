package tv.floe.metronome.clustering.kmeans;

import static org.junit.Assert.*;

import org.junit.Test;

import tv.floe.metronome.clustering.kmeans.DistanceMeasurer;
import tv.floe.metronome.clustering.kmeans.Point;

public class DistanceMeasurerTest {

	@Test
	public void test1() {
		double [] d1 = {0,0};
		double [] d2 = {3,4};
		Point p1 = new Point(d1);
		Point p2 = new Point(d2);
		
		DistanceMeasurer measurer = new DistanceMeasurer();
		assertEquals(5.0, measurer.distance(p1, p2), 0.0000000001);
		assertEquals(5.0, measurer.distance(p2, p1), 0.0000000001);
	}

	@Test
	public void test2() {
		double [] d1 = {-2,-1};
		double [] d2 = {1,3};
		Point p1 = new Point(d1);
		Point p2 = new Point(d2);
		
		DistanceMeasurer measurer = new DistanceMeasurer();
		assertEquals(5.0, measurer.distance(p1, p2), 0.0000000001);
		assertEquals(5.0, measurer.distance(p2, p1), 0.0000000001);
	}
	
}
