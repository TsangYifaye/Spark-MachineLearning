import globalVal
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.stat import MultivariateStatisticalSummary
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.feature import StandardScalerModel,StandardScaler
from pyspark.mllib.regression import LabeledPoint

def TrainDTModel(trainData, impurity, maxDepth, maxBins):   # Decision Tree
    model = DecisionTree.trainClassifier(data = trainData, numClasses = 2, categoricalFeaturesInfo = {}, \
                                         impurity = impurity, maxDepth = maxDepth, maxBins = maxBins)
    return model

def TrainLRModel(trainData, iterations, step, miniBatchFraction):   # Logistic Regression
    srcFeatures = trainData.map(lambda line: line.features)
    print srcFeatures.first()
    scaler = StandardScaler(withMean = True, withStd = True).fit(srcFeatures)
    srcLabel = trainData.map(lambda line: line.label)
    scaledFeature = scaler.transform(srcFeatures)
    print scaledFeature.first()
    scaledData = srcLabel.zip(scaledFeature)
    trainData = scaledData.map(lambda (label, features): LabeledPoint(label, features))
    model = LogisticRegressionWithSGD.train(data = trainData, iterations = iterations, step = step, \
                                            miniBatchFraction = miniBatchFraction)
    return model

def EvaluateModel(model, validationData):
    # Python version of the DecisionTreeModel can't handle currently equivalent of
    # "validationData.map(lambda point: (model.predict(point.features), point.label))" but Scala can.Hence I use zip() instead
    # scoresAndLabels = validationData.map(lambda point: (model.predict(point.features), point.label))
    # metrics = BinaryClassificationMetrics(scoresAndLabels)
    if(model.__class__.__name__ == "DecisionTreeModel"):
        predictedLabel = model.predict(validationData.map(lambda line: line.features))
        scoresAndLabels = validationData.map(lambda line: line.label).zip(predictedLabel)
        areaUnderROC = float(BinaryClassificationMetrics(scoresAndLabels).areaUnderROC)
        matchedNum = scoresAndLabels.filter(lambda (real, predicted): real == predicted).count()
        accRate = float(matchedNum) / validationData.count()
    else :
        scoresAndLabels = validationData.map(lambda line: (model.predict(line.features), line.label)).collect()
        scoresAndLabels = [[float(i), j] for i, j in scoresAndLabels]
        rdd_scoresAndLabels = globalVal.sc.parallelize(scoresAndLabels)
        areaUnderROC = float(BinaryClassificationMetrics(rdd_scoresAndLabels).areaUnderROC)
        matchedNum = rdd_scoresAndLabels.filter(lambda (real, predicted): real == predicted).count()
        accRate = float(matchedNum) / validationData.count()
        # matchedNum = validationData.map(lambda line: 1 if (model.predict(line.features) == line.label) else 0 ).sum()
        # accRate = float(matchedNum) / validationData.count()
    return areaUnderROC, accRate