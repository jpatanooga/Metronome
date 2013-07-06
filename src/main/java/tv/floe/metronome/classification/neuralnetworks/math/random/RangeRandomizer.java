package tv.floe.metronome.classification.neuralnetworks.math.random;

public class RangeRandomizer  extends WeightsRandomizer {
    /**
     * Lower range limit
     */
    protected double min;
    
    /**
     * Upper range limit
     */
    protected double max;

    /**
     * Creates a new instance of RangeRandomizer within specified .
     * The random values are generated according to formula:
     * newValue = min + random * (max - min)
     * @param min min weight value
     * @param max max weight value
     */
    public RangeRandomizer(double min, double max) {
        this.max = max;
        this.min = min;
    }

    /**
     * Generates next random value within [min, max] range determined by the settings in this randomizer
     * @return next weight random value
     */
    @Override
    protected double nextRandomWeight() {
        return min + randomGenerator.nextDouble() * (max - min);
    }
}
