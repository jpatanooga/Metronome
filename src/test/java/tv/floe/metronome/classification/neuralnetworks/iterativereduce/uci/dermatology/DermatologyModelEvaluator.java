package tv.floe.metronome.classification.neuralnetworks.iterativereduce.uci.dermatology;

import tv.floe.metronome.classification.neuralnetworks.eval.ModelEvaluator;

public class DermatologyModelEvaluator {

	public static void main(String[] args) throws Exception {
		
		ModelEvaluator.eval("src/test/resources/run_profiles/unit_tests/nn/dermatology/app.unit_test.nn.dermatology.properties");
		
	}

}
