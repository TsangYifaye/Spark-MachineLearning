import globalVal
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors

def LoadData():
    rawDataWithHeader = globalVal.sc.textFile("../data/train.tsv")
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda line : line != header)
    data = rawData.map(lambda line: line.split('\t'))
    return data

def ThirdFeatureProcess(data,categoriesDict,categoriesNum):
    categoriesVector = [0]*categoriesNum
    categoriesVector[categoriesDict[data]] = 1
    return categoriesVector

def TransformData(data):
    trimmed = data.map(lambda line: [unit.replace('\"', '') for unit in line])
    categoriesDict = dict(trimmed.map(lambda line: line[3]).distinct().zipWithIndex().collect())
    categoriesNum = len(categoriesDict)

    labeledPointData = trimmed.map(lambda line: (line[-1], line[3], line[4:-1])). \
        map(lambda (label, thirdFeature, otherFeatures): (int(label), ThirdFeatureProcess(thirdFeature,categoriesDict,categoriesNum), \
                                                          [0.0 if unit == '?' else float(unit) for unit in otherFeatures])). \
        map(lambda (label, thirdFeature, otherFeatures): LabeledPoint(label, Vectors.dense(thirdFeature + otherFeatures)))
    return labeledPointData

def SplitData(labeledPointData):
    trainData, validationData, testData = labeledPointData.randomSplit([0.8, 0.1, 0.1])
    return trainData, validationData, testData

def PrepareData():
    data = LoadData()
    labeledPointData = TransformData(data)
    trainData, validationData, testData = SplitData(labeledPointData)
    return trainData, validationData, testData
