package tv.floe.metronome.classification.neuralnetworks.core.neurons;

import tv.floe.metronome.classification.neuralnetworks.input.WeightedSum;
import tv.floe.metronome.classification.neuralnetworks.transfer.Linear;

public class InputNeuron extends Neuron {

    public InputNeuron() {
        super(new WeightedSum(), new Linear());
    }

    @Override
    public void calcOutput() {
        this.output = this.netInput;
    }
}