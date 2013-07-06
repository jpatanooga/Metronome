package tv.floe.metronome.classification.neuralnetworks.transfer;

public enum TransferFunctionType {

	LINEAR("Linear"),
	RAMP("Ramp"),
	STEP("Step"),
	SIGMOID("Sigmoid"),
	TANH("Tanh"),
//	GAUSSIAN("Gaussian"),
//	TRAPEZOID("Trapezoid"),
//	SGN("Sgn"), 
//        SIN("Sin"), 
        LOG("Log");

		private String typeLabel;
		
		private TransferFunctionType(String typeLabel) {
			this.typeLabel = typeLabel;
		}
		
		public String getTypeLabel() {
			return typeLabel;
		}

	        public Class getTypeClass() {
	            switch (this) {
	                case LINEAR:
				return Linear.class;
			case STEP:
				return Step.class;
			case RAMP:
				return Ramp.class;
			case SIGMOID:
				return Sigmoid.class;
			case TANH:
				return Tanh.class;
/*			case TRAPEZOID:
				return Trapezoid.class;
			case GAUSSIAN:
				return Gaussian.class;
			case SGN:
				return Sgn.class; 
			case SIN:
				return Sin.class;                     
			case LOG:
				return Log.class;  
				*/                   
			} // switch

            return null;

            }

	  
}
