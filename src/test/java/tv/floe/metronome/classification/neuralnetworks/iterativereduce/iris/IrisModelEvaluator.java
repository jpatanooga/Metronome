package tv.floe.metronome.classification.neuralnetworks.iterativereduce.iris;

import tv.floe.metronome.classification.neuralnetworks.eval.ModelEvaluator;

public class IrisModelEvaluator {

	public static void main(String[] args) throws Exception {

		ModelEvaluator.eval("src/test/resources/run_profiles/unit_tests/nn/iris/app.unit_test.nn.iris.properties");

	}

}
