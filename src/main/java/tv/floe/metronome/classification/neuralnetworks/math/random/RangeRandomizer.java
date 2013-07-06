package tv.floe.metronome.classification.neuralnetworks.math.random;

public class RangeRandomizer  extends WeightsRandomizer {
    protected double min;
    protected double max;

    public RangeRandomizer(double min, double max) {
        this.max = max;
        this.min = min;
    }

    @Override
    protected double nextRandomWeight() {
        return min + randomGenerator.nextDouble() * (max - min);
    }
}
