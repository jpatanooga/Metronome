package tv.floe.metronome.classification.neuralnetworks.conf;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.core.Weight;
import tv.floe.metronome.classification.neuralnetworks.input.InputFunction;

public class TestConfig {
	
	@Test
	public void testSetGetConf( ) throws InstantiationException, IllegalAccessException {
		
		Config c = new Config();
		
		c.parse(null);
		
		Class inFuncClass = (Class) c.getConfValue("inputFunction");
		InputFunction f = (InputFunction) inFuncClass.newInstance();
		
		
		
		assertEquals("tv.floe.metronome.classification.neuralnetworks.input.WeightedSum", f.getClass().getName() );
		
		System.out.println("> f: " + f.getClass().getName());
		
		
		
	}

}
