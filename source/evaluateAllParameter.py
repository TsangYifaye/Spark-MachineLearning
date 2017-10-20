import globalVal
from trainEvaluateModel import *

def EvaluateAllParameter(trainData, validationData, impuritySet, maxDepthSet, maxBinsSet):
    candidateModelList = []
    for impurity in impuritySet:
        for maxDepth in maxDepthSet:
            for maxBins in maxBinsSet:
                model = TrainDTModel(trainData, impurity, maxDepth, maxBins)
                areaUnderROC , accRate = EvaluateModel(model, validationData)
                print areaUnderROC, accRate
                print "Area under ROC: %2.4f%%, Accuracy : %2.4f%%, Impurity: %s, MaxDepth: %d, MaxBins: %d" % \
                      ((areaUnderROC * 100), (accRate * 100), impurity, maxDepth, maxBins)
                candidateModelList.append((model, areaUnderROC, accRate, impurity, maxDepth, maxBins))
    candidateModelList = sorted(candidateModelList, key = lambda candidateModel: candidateModel[1], reverse=True)  #sort by ROC
    return candidateModelList[0][0]