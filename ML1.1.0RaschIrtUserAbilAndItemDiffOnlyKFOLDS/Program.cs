using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;

namespace ML1._1._0RaschIrtUserAbilAndItemDiffOnlyKFOLDS
{
    class Program
    {
        //tutorial to use this version
        //https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/sentiment-analysis

        //tutorial for K-folds
        //https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/train-machine-learning-model-cross-validation-ml-net

        //needed this to show how to load multipel features and concatenate
        //https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/github-issue-classification

        static readonly string _envCurr = Environment.CurrentDirectory;
        static readonly string _dataPathNew = Directory.GetParent(_envCurr).FullName;



        static readonly string _dataPathNewAgain = Directory.GetParent(_dataPathNew).Parent.FullName;
        static readonly string _dataPath = Path.Combine(_dataPathNewAgain, "Data", "KFoldDatausing1.00etcJustIrtStuff.txt");

        //where the model will be saved when created, but this does not exist until data has been trained on

        //static readonly string _modelPath = Path.Combine(_dataPathNewAgain, "Data", "ModelStochGradDescOptionsSet.zip");

        //static string AlgorithmUsed;

        static void Main(string[] args)
        {

            MLContext mlContext = new MLContext();
            //now gets standard data returned, not split data
            var splitDataView = LoadData(mlContext);
            //variable below is now not just a model, but a model for each fol, train/test data for each fold, metrcis object for each fold
            var model = BuildAndTrain(mlContext, splitDataView);
            Evaluate(mlContext, model, splitDataView);
            //UseModelWithSingleItem(mlContext, model);
            Console.ReadLine();

        }


        private static IDataView LoadData(MLContext mlContext)
        {
            //loads the data only
            IDataView dataView = mlContext.Data.LoadFromTextFile<QuestionData>(_dataPath, hasHeader: false);
            //basic splitting of data, no K-folds yet
            //TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);


            return dataView;
        }

        //calibrated version below
        // public static IReadOnlyList<TrainCatalogBase.CrossValidationResult<CalibratedBinaryClassificationMetrics>> BuildAndTrain(MLContext mlContext, IDataView splitTrainSet)

        public static IReadOnlyList<TrainCatalogBase.CrossValidationResult<BinaryClassificationMetrics>> BuildAndTrain(MLContext mlContext, IDataView splitTrainSet)
        {
            //var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(QuestionData.QuestionText))
                       

            //transformedData used to be pipeline when algorithm was in it also
            IEstimator<ITransformer> transformedDataStage1 = mlContext.Transforms.Text.FeaturizeText(inputColumnName: "UserAbility", outputColumnName: "UserAbilityFeaturized")

                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "QuestionDifficulty", outputColumnName: "QuestionDifficultyFeaturized")
                .Append(mlContext.Transforms.Concatenate("Features", "UserAbilityFeaturized", "QuestionDifficultyFeaturized")));

            //what does the fit method do? These two methods involve transformign teh data for use in the algorithm, unspecific though
            var dataPrepTransformer = transformedDataStage1.Fit(splitTrainSet);
            IDataView transformedDataStage2 = dataPrepTransformer.Transform(splitTrainSet);

            //LogReg Stopchastic used as this was given in the sample, which used calibrated model, will change to SVM

            //IEstimator<ITransformer> svmLinAlg = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression();
            //var cvResults = mlContext.BinaryClassification.CrossValidate(transformedDataStage2, svmLinAlg, numberOfFolds: 10);


            //svm linear now used

            IEstimator<ITransformer> svmLinAlg = mlContext.BinaryClassification.Trainers.LinearSvm();
            var cvResults = mlContext.BinaryClassification.CrossValidateNonCalibrated(transformedDataStage2, svmLinAlg, numberOfFolds: 10);


            Console.WriteLine("=============== Create and Train the Model ===============");
           
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            //the cvResults object will contain a lot of things:

           // 1. TrainTestData object for each fold of data
           // 2. a model for each fold
           // 3. a metric for each fold
        
            return cvResults;


        }

        //calibrated version below
        //public static void Evaluate(MLContext mlContext, IReadOnlyList<TrainCatalogBase.CrossValidationResult<CalibratedBinaryClassificationMetrics>> cvResults, IDataView splitTestSet)

        public static void Evaluate(MLContext mlContext, IReadOnlyList<TrainCatalogBase.CrossValidationResult<BinaryClassificationMetrics>> cvResults, IDataView splitTestSet)
        {

            //model contains these things
            // 1. TrainTestData object for each fold of data
            // 2. a model for each fold
            // 3. a metric for each fold


            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");


            IEnumerable<double> rSquared =  cvResults
                .Select(fold => fold.Metrics.Accuracy);
            ITransformer[] models = cvResults
                .OrderByDescending(fold => fold.Metrics.Accuracy)
                .Select(fold => fold.Model)
                .ToArray();
            ITransformer topModel = models[0];

            foreach(double acc in rSquared)
            {
                Console.WriteLine(acc.ToString());
            }

            Console.WriteLine($"Higest Accuracy: {topModel.ToString()}");



            //IDataView predictions = model.Transform(splitTestSet);

            //Below for when calibrated is used
            //CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            //Below for noncalibrated - probability not generated with certain models, e.g. svmlinear
            //BinaryClassificationMetrics metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(predictions, "Label");

            //below metrics for regression algorithms
            //var metrics = mlContext.Regression.Evaluate(predictions, "Label");




            //Console.WriteLine();
            //Console.WriteLine("Model quality metrics evaluation");
            //Console.WriteLine("--------------------------------");
            ////below for when regression algorithm used as other metrics not available
            ////Console.WriteLine($"yes {metrics.RSquared}");
            //Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            //Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            //Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            //Console.WriteLine("=============== End of model evaluation ===============");

        }

        public static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {

            PredictionEngine<QuestionData, QuestionPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<QuestionData, QuestionPrediction>(model);

            QuestionData sampleStatement = new QuestionData
            {
                UserAbility = "0.57",
                QuestionDifficulty = "0.51"
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();

            //below for when calibrated metrics are used - for probability column
            //Console.WriteLine($"User Ability: {sampleStatement.UserAbility}  | Item Difficulty | {sampleStatement.QuestionDifficulty} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Correct" : "Incorrect")} | Probability: {resultPrediction.Probability} ");

            //below for when probability is not used
            Console.WriteLine($"Algorithm Used: Linear SVM | User Ability: {sampleStatement.UserAbility}  | Item Difficulty | {sampleStatement.QuestionDifficulty} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Correct" : "Incorrect")}");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();

        }
    }
}
