package weka.api;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Discretize;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

public class HelloWeka {
	
	public static Instances filterNumericToNominal(Instances dataset, String[] opts) throws Exception {
		NumericToNominal numToNom = new NumericToNominal();
		numToNom.setOptions(opts);
		numToNom.setInputFormat(dataset);
		Instances nomData = Filter.useFilter(dataset, numToNom);
		return nomData;
	}
	
	public static Instances filterDiscretize(Instances dataset, String[] opts) throws Exception {
		Discretize discretize = new Discretize();
		discretize.setOptions(opts);
		discretize.setInputFormat(dataset);
		Instances numData = Filter.useFilter(dataset, discretize);
		return numData;
	}
	
	public static NaiveBayes buldNaiveBayesWithCrossValidation(Instances dataset, int numFold) throws Exception {
		dataset.setClassIndex(dataset.numAttributes() - 1);
		NaiveBayes nb = new NaiveBayes();
		int seed = 1;
		int folds = 15;
		Random rand = new Random(seed);
		Instances randData = new Instances(dataset);
		randData.randomize(rand);
		
		//stratify	    
		if (randData.classAttribute().isNominal())
			randData.stratify(folds);		
		
		for (int n = 0; n < folds; n++) {
			Evaluation eval = new Evaluation(randData);
			//get the folds	      
			Instances train = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);	      
			// build and evaluate classifier	     
			nb.buildClassifier(train);
			//eval.evaluateModel(nb, test);

			// output evaluation
			System.out.println();
			System.out.println(eval.toMatrixString("=== Confusion matrix for fold " + (n+1) + "/" + folds + " ===\n"));
			System.out.println("Correct % = "+eval.pctCorrect());
			System.out.println("Incorrect % = "+eval.pctIncorrect());
			System.out.println("AUC = "+eval.areaUnderROC(1));
			System.out.println("kappa = "+eval.kappa());
			System.out.println("MAE = "+eval.meanAbsoluteError());
			System.out.println("RMSE = "+eval.rootMeanSquaredError());
			System.out.println("RAE = "+eval.relativeAbsoluteError());
			System.out.println("RRSE = "+eval.rootRelativeSquaredError());
			System.out.println("Precision = "+eval.precision(1));
			System.out.println("Recall = "+eval.recall(1));
			System.out.println("fMeasure = "+eval.fMeasure(1));
			System.out.println("Error Rate = "+eval.errorRate());
			//the confusion matrix
			//System.out.println(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));
		}
		return nb;
	}
	
	public static NaiveBayes buildNaiveBayesWithFullTraining(Instances dataset) throws Exception {
		dataset.setClassIndex(dataset.numAttributes()-1);
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(dataset);
		return nb;
	}
	
	
	public static void saveNaiveBayesModel(String model_name, NaiveBayes nb) throws Exception {
		weka.core.SerializationHelper.write(model_name, nb);
	}
	
	public static NaiveBayes loadNaiveBayesModel(String model_name) throws Exception {
		NaiveBayes nbLoad = (NaiveBayes) weka.core.SerializationHelper.read("my_nb_model.model");
		return nbLoad;
		
	}
	
	public static void inputNewInstance(Instances dataset, double attr1, double attr2, double attr3, double attr4, double attr5) throws Exception {
		double[] instanceValue = new double[dataset.numAttributes()];
		instanceValue[0] = attr1;
        instanceValue[1] = attr2;
        instanceValue[2] = attr3;
        instanceValue[3] = attr4;
        instanceValue[4] = attr5;
        dataset.add(new DenseInstance(1,instanceValue));
        System.out.println(dataset);
        
		
	}

	
	public static void main (String[] args) throws Exception {
		//Read Data Set and assign to dataset
		DataSource source = new DataSource("/home/cmrudi/weka-3-8-0/data/iris.arff");
		Instances dataset = source.getDataSet();
		
		System.out.println(dataset);
		
		
		//Use NumericToNominal and assign to nomData
		String[] opts = new String[]{"-R","1,2"};
		Instances nomData = filterNumericToNominal(dataset,opts);
		
		
		//Use Discretize and and assign to numData
		String[] discOpts = new String[]{"-B","4","-R","3"};
		Instances numData = filterDiscretize(dataset,discOpts);
		
		
		//Use Naive Bayes with 10 Fold Cross Validation
		//Belum Berhasil
		
		//Skema Full Training using Naive Bayes
		NaiveBayes nb = buildNaiveBayesWithFullTraining(dataset);
		
		
		//Save Model
		saveNaiveBayesModel("my_nb_model.model", nb);
		
		//Load Model
		NaiveBayes newNb = loadNaiveBayesModel("my_nb_model.model");
		
		//inputInstance
		inputNewInstance(dataset,5.8,2.9,5,1.7,2);
	}
}
