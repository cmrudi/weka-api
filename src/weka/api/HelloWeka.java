package weka.api;

import java.io.File;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Discretize;

public class HelloWeka {
	public static void main (String[] args) throws Exception {
		//Read Data Set and assign to dataset
		DataSource source = new DataSource("/home/cmrudi/weka-3-8-0/data/iris.arff");
		Instances dataset = source.getDataSet();
		System.out.println(dataset.toSummaryString());
		
		//Use NumericToNominal and assign to nomData
		String[] opts = new String[]{"-R","1,2"};
		NumericToNominal numToNom = new NumericToNominal();
		numToNom.setOptions(opts);
		numToNom.setInputFormat(dataset);
		Instances nomData = Filter.useFilter(dataset, numToNom);
		System.out.println(nomData.toSummaryString());
		
		//Use Discretize and and assign to numData
		String[] discOpts = new String[]{"-B","4","-R","3"};
		Discretize discretize = new Discretize();
		discretize.setOptions(discOpts);
		discretize.setInputFormat(nomData);
		Instances numData = Filter.useFilter(nomData, discretize);
		System.out.println(numData.toSummaryString());
		
		
		
	}
}
