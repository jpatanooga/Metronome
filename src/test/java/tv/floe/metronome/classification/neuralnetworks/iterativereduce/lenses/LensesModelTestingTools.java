package tv.floe.metronome.classification.neuralnetworks.iterativereduce.lenses;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.eval.ModelEvaluator;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;
import tv.floe.metronome.io.records.MetronomeRecordFactory;

public class LensesModelTestingTools {
	
	public static void main(String[] args) throws Exception {
		
		ModelEvaluator.eval("src/test/resources/run_profiles/unit_tests/nn/lenses/app.unit_test.nn.lenses.properties");
		
	}
	
}
