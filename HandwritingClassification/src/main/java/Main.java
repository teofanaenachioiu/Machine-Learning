import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        int batchSize = 16; // how many examples to simultaneously train in the network
        EmnistDataSetIterator.Set emnistSet = EmnistDataSetIterator.Set.BALANCED;
        EmnistDataSetIterator emnistTrain = new EmnistDataSetIterator(emnistSet, batchSize, true);
        EmnistDataSetIterator emnistTest = new EmnistDataSetIterator(emnistSet, batchSize, false);

        int outputNum = EmnistDataSetIterator.numLabels(emnistSet); // total output classes
        int rngSeed = 123; // integer for reproducability of a random number generator
        int numRows = 28; // number of "pixel rows" in an mnist digit
        int numColumns = 28;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numRows * numColumns) // Number of input datapoints.
                        .nOut(1000) // Number of output datapoints.
                        .activation(Activation.RELU) // Activation function.
                        .weightInit(WeightInit.XAVIER) // Weight initialization.
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
//                .pretrain(false)
//                .backprop(true)
                .build();

//      create the MLN
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

//      pass a training listener that reports score every 10 iterations
        int eachIterations = 5;
        network.addListeners(new ScoreIterationListener(eachIterations));

//      fit a dataset for a single epoch
//        network.fit(emnistTrain);

//      fit for multiple epochs
        int numEpochs = 5;
        network.fit(new MultipleEpochsIterator(numEpochs, emnistTrain));

//      evaluate basic performance
        Evaluation eval = network.evaluate(emnistTest);
        eval.accuracy();
        eval.precision();
        eval.recall();

//      evaluate ROC and calculate the Area Under Curve
//        ROCMultiClass roc = network.evaluateROCMultiClass(emnistTest);
//        roc.calculateAverageAUC();

//        int classIndex = 0;
//        roc.calculateAUC(classIndex);

//      optionally, you can print all stats from the evaluations
        System.out.println(eval.stats(false,true));
//        System.out.println(roc.stats());

//        System.out.println( System.getProperty("os.arch"));
    }


}
