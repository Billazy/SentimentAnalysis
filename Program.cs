// <SnippetAddUsings>
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
// </SnippetAddUsings>

namespace SentimentAnalysis
{
    class Program
    {
        // <SnippetDeclareGlobalVariables>
        // Dataset utilisé : Amazon (format TSV, sans en-tête)
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "amazon_cells_labelled.txt");
        // </SnippetDeclareGlobalVariables>

        // Chemin du modèle sauvegardé
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "sentiment_model.zip");

        // Seed fixe pour rendre tous les runs reproductibles
        private const int Seed = 42;

        // Nombre de folds pour la validation croisée
        private const int NumberOfFolds = 10;

        static void Main()
        {
            // <SnippetCreateMLContext>
            // Seed fixe : mêmes folds, mêmes résultats à chaque exécution
            MLContext mlContext = new(seed: Seed);
            // </SnippetCreateMLContext>

            // ============================================================
            // 1) VALIDATION CROISÉE 10-FOLD : comparaison fiable des configs
            // ============================================================
            Console.WriteLine("############################################################");
            Console.WriteLine("### VALIDATION CROISÉE 10-FOLD SUR LE DATASET AMAZON     ###");
            Console.WriteLine("############################################################");

            var cvA = CrossValidateConfig(mlContext, "CONFIG A : bigrammes + 3-grammes caractères", useCharFeatures: true);
            var cvB = CrossValidateConfig(mlContext, "CONFIG B : bigrammes de mots uniquement", useCharFeatures: false);

            // ============================================================
            // 2) TEST T APPARIÉ : la différence A vs B est-elle significative ?
            // ============================================================
            Console.WriteLine();
            Console.WriteLine("############################################################");
            Console.WriteLine("### TEST STATISTIQUE APPARIÉ (A vs B)                    ###");
            Console.WriteLine("############################################################");

            double pValue = PairedTTest(cvA.FoldAccuracies, cvB.FoldAccuracies, out double tStat);
            Console.WriteLine($"t = {tStat:F3}, p ≈ {pValue:F3}");

            // Décision : on retient A si elle est meilleure ET si p < 0.05
            bool useCharFeatures;
            string chosenConfig;
            if (cvA.MeanAccuracy > cvB.MeanAccuracy && pValue < 0.05)
            {
                useCharFeatures = true;
                chosenConfig = "CONFIG A (bigrammes + char-ngrams) — gain significatif";
            }
            else if (cvB.MeanAccuracy > cvA.MeanAccuracy && pValue < 0.05)
            {
                useCharFeatures = false;
                chosenConfig = "CONFIG B (bigrammes seuls) — gain significatif";
            }
            else
            {
                // Pas de différence significative : on prend la plus simple (B)
                useCharFeatures = false;
                chosenConfig = "CONFIG B (bigrammes seuls) — différence non significative, simplicité retenue";
            }

            Console.WriteLine();
            Console.WriteLine($"Config retenue : {chosenConfig}");
            Console.WriteLine($"useCharFeatures = {useCharFeatures}");

            // ============================================================
            // 3) PIPELINE FINAL COMPLET (config retenue) + sauvegarde
            // ============================================================
            Console.WriteLine();
            Console.WriteLine("############################################################");
            Console.WriteLine("### PIPELINE FINAL COMPLET + SAUVEGARDE                  ###");
            Console.WriteLine("############################################################");

            TrainTestData splitDataView = LoadData(mlContext);
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet, useCharFeatures);
            Evaluate(mlContext, model, splitDataView.TestSet);
            AnalyzeThresholds(mlContext, model, splitDataView.TestSet);
            SaveModel(mlContext, model, splitDataView.TrainSet);
            UseModelWithSingleItem(mlContext, model);
            UseModelWithBatchItems(mlContext, model);
            UseLoadedModel(mlContext);

            Console.WriteLine();
            Console.WriteLine("=============== End of process ===============");
        }

        /// <summary>
        /// Résultat d'une validation croisée : moyennes + accuracies par fold.
        /// </summary>
        private record CrossValidationResult(
            double MeanAccuracy,
            double StdAccuracy,
            double MeanAuc,
            double MeanF1,
            double[] FoldAccuracies);

        /// <summary>
        /// Construit le pipeline de featurisation + entraîneur.
        /// Point unique de construction, réutilisé par la CV et le pipeline final.
        /// </summary>
        private static IEstimator<ITransformer> BuildPipeline(MLContext mlContext, bool useCharFeatures)
        {
            TextFeaturizingEstimator.Options options = new()
            {
                WordFeatureExtractor = new WordBagEstimator.Options
                {
                    NgramLength = 2,
                    UseAllLengths = true
                }
            };

            if (useCharFeatures)
            {
                options.CharFeatureExtractor = new WordBagEstimator.Options
                {
                    NgramLength = 3,
                    UseAllLengths = false
                };
            }

            return mlContext.Transforms.Text.FeaturizeText(
                    outputColumnName: "Features",
                    options: options,
                    inputColumnNames: new[] { nameof(SentimentData.SentimentText) })
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "Label",
                    featureColumnName: "Features"));
        }

        /// <summary>
        /// Validation croisée 10-fold : entraîne 10 modèles sur 10 splits différents
        /// et moyenne les métriques.
        /// </summary>
        private static CrossValidationResult CrossValidateConfig(
            MLContext mlContext, string configName, bool useCharFeatures)
        {
            Console.WriteLine();
            Console.WriteLine($"========== {configName} ==========");

            IDataView fullData = mlContext.Data.LoadFromTextFile<SentimentData>(
                _dataPath, hasHeader: false, separatorChar: '\t');

            var pipeline = BuildPipeline(mlContext, useCharFeatures);

            // 10-fold : chaque fold sert une fois de test
            var cvResults = mlContext.BinaryClassification.CrossValidate(
                fullData, pipeline, numberOfFolds: NumberOfFolds, labelColumnName: "Label");

            var accuracies = cvResults.Select(r => r.Metrics.Accuracy).ToArray();
            double meanAccuracy = accuracies.Average();
            double stdAccuracy = Math.Sqrt(
                accuracies.Average(a => Math.Pow(a - meanAccuracy, 2)));
            double meanAuc = cvResults.Average(r => r.Metrics.AreaUnderRocCurve);
            double meanF1 = cvResults.Average(r => r.Metrics.F1Score);

            Console.WriteLine($"Accuracy moyenne : {meanAccuracy:P2} (± {stdAccuracy:P2})");
            Console.WriteLine($"AUC moyen        : {meanAuc:P2}");
            Console.WriteLine($"F1 moyen         : {meanF1:P2}");
            Console.WriteLine("Accuracies par fold : " +
                string.Join(", ", accuracies.Select(a => a.ToString("P2"))));
            Console.WriteLine("=================================================");

            return new CrossValidationResult(meanAccuracy, stdAccuracy, meanAuc, meanF1, accuracies);
        }

        /// <summary>
        /// Test t apparié sur les accuracies par fold.
        /// Seuils critiques bilatéraux pour df = 9 (10 folds) :
        ///   p=0.10 -> 1.833 | p=0.05 -> 2.262 | p=0.01 -> 3.250
        /// </summary>
        private static double PairedTTest(double[] scoresA, double[] scoresB, out double tStat)
        {
            var diffs = scoresA.Zip(scoresB, (a, b) => a - b).ToArray();
            int n = diffs.Length;
            double meanDiff = diffs.Average();
            double stdDiff = Math.Sqrt(diffs.Average(d => Math.Pow(d - meanDiff, 2)));

            tStat = stdDiff == 0 ? 0 : meanDiff / (stdDiff / Math.Sqrt(n));

            double absT = Math.Abs(tStat);
            return absT switch
            {
                >= 3.250 => 0.01,
                >= 2.262 => 0.05,
                >= 1.833 => 0.10,
                _ => 0.50
            };
        }

        public static TrainTestData LoadData(MLContext mlContext)
        {
            // <SnippetLoadData>
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(
                _dataPath,
                hasHeader: false,
                separatorChar: '\t');
            // </SnippetLoadData>

            // <SnippetSplitData>
            // Seed fixe également pour le split final (reproductibilité)
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(
                dataView, testFraction: 0.2, samplingKeyColumnName: null, seed: Seed);
            // </SnippetSplitData>

            // <SnippetReturnSplitData>
            return splitDataView;
            // </SnippetReturnSplitData>
        }

        /// <summary>
        /// Construit et entraîne le modèle.
        /// </summary>
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet, bool useCharFeatures)
        {
            var estimator = BuildPipeline(mlContext, useCharFeatures);

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);

            CalibratedBinaryClassificationMetrics metrics =
                mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        /// <summary>
        /// Teste plusieurs seuils de décision sur le jeu de test.
        /// </summary>
        private static void AnalyzeThresholds(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine();
            Console.WriteLine("=============== Threshold analysis ===============");

            IDataView scoredData = model.Transform(splitTestSet);

            var predictions = mlContext.Data
                .CreateEnumerable<SentimentPrediction>(scoredData, reuseRowObject: false)
                .ToList();

            if (predictions.Count == 0)
            {
                Console.WriteLine("Jeu de test vide, impossible d'analyser les seuils.");
                return;
            }

            foreach (double threshold in new double[] { 0.3, 0.4, 0.5, 0.6, 0.7 })
            {
                int correct = 0;
                foreach (var p in predictions)
                {
                    double probability = 1.0 / (1.0 + Math.Exp(-p.Score));
                    bool predictedLabel = probability > threshold;
                    if (predictedLabel == p.Sentiment)
                        correct++;
                }

                double accuracy = (double)correct / predictions.Count;
                Console.WriteLine($"Seuil {threshold:F1} : Accuracy = {accuracy:P2}");
            }

            Console.WriteLine("=============== End of threshold analysis ===============");
        }

        private static void SaveModel(MLContext mlContext, ITransformer model, IDataView trainSet)
        {
            Console.WriteLine();
            Console.WriteLine("=============== Saving model ===============");
            mlContext.Model.Save(model, trainSet.Schema, _modelPath);
            Console.WriteLine($"Modèle sauvegardé dans : {_modelPath}");
            Console.WriteLine("=============== End of saving ===============");
        }

        private static void UseLoadedModel(MLContext mlContext)
        {
            Console.WriteLine();
            Console.WriteLine("=============== Loading and using saved model ===============");

            if (!File.Exists(_modelPath))
            {
                Console.WriteLine($"Fichier introuvable : {_modelPath}");
                return;
            }

            ITransformer loadedModel = mlContext.Model.Load(_modelPath, out _);

            var engine = mlContext.Model
                .CreatePredictionEngine<SentimentData, SentimentPrediction>(loadedModel);

            var result = engine.Predict(new SentimentData { SentimentText = "This is great!" });

            Console.WriteLine($"Sentiment: {result.SentimentText} | " +
                              $"Prediction: {(Convert.ToBoolean(result.Prediction) ? "Positive" : "Negative")} | " +
                              $"Probability: {result.Probability}");

            Console.WriteLine("=============== End of loaded model test ===============");
        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction =
                mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            SentimentData sampleStatement = new()
            {
                SentimentText = "This was a very bad steak"
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");
            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            IEnumerable<SentimentData> sentiments =
            [
                new() { SentimentText = "This was a horrible meal" },
                new() { SentimentText = "I love this spaghetti." }
            ];

            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);
            IDataView predictions = model.Transform(batchComments);

            IEnumerable<SentimentPrediction> predictedResults =
                mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            Console.WriteLine();

            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}