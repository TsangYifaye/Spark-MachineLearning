import pyspark
import globalVal
import prepareData
import trainEvaluateModel
from evaluateAllParameter import EvaluateAllParameter
from prepareData import PrepareData
from trainEvaluateModel import *

conf = pyspark.SparkConf().setAppName("DecisionTree").setMaster("local[2]")
globalVal.sc = pyspark.SparkContext(conf = conf)

trainData, validationData, testData = PrepareData()
# model = TrainModel(trainData, "gini", 20, 10)
# EvaluateModel(model, validationData)

#bestModel = EvaluateAllParameter(trainData,validationData,("gini","entropy"),(3, 5, 10, 15, 20),(3, 5, 10, 15, 20, 25, 30, 32))
bestModel = TrainLRModel(trainData, 50, 1, 1)
areaUnderROC, accRate = EvaluateModel(bestModel, trainData)
print "TestData Area under ROC: %2.4f%%, Accuracy : %2.4f%%" % ((areaUnderROC * 100), (accRate * 100))