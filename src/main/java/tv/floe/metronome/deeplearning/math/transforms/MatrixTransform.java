package tv.floe.metronome.deeplearning.math.transforms;

import java.io.Serializable;

import org.apache.mahout.math.Matrix;


import com.google.common.base.Function;


public interface MatrixTransform extends Function<Matrix, Matrix>,Serializable {

}
