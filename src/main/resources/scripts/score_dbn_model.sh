script_name=$0
properties_file=$1


java -cp `yarn classpath`:./Metronome-1.0-SNAPSHOT-jar-with-dependencies.jar tv.floe.metronome.deeplearning.dbn.model.evaluation.YarnCmdLineModelTester $properties_file

