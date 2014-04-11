package tv.floe.metronome.deeplearning.neuralnetwork.activation;


public class Activations {

	public static ActivationFunction tanh() {
		return new TanH();
	}
	
	public static ActivationFunction sigmoid() {
		return new Sigmoid();
	}

	
	public static ActivationFunction hardTanh() {
		return new HardTanh();
	}
	
	
	
	public static ActivationFunction softmax() {
		return new SoftMax();
	}
}
