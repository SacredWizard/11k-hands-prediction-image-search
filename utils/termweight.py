from numpy import argsort

def getDataLatentSemantics(data_M, k):
    term = [argsort(-data_M[:, i]) for i in range(k)]
    TW=[]
    #for each latent semantic (k)
    for i in range(k):
        kTW = []
        #for each data object
        for j in range(len(term[i])):
            kTW.append([term[i][j],data_M[term[i][j],i]])
        TW.append(kTW)
    return TW

def getFeatureLatentSemantics(feature_M, k):
    term = [argsort(-feature_M[i, :]) for i in range(k)]
    TW=[]
    #for each latent semantic (k)
    for i in range(k):
        kTW = []
        #for each feature
        for j in range(len(term[i])):
            kTW.append([term[i][j],feature_M[i,term[i][j]]])
        TW.append(kTW)
    return TW

    #prints Term Weight Pairs to console
def printTW(data_TW, feature_TW, feature_descriptor):
    print()
    print("-------------------------------------------------------------------------------------------------------------")
    print()
    print("Data Latent semantics\n")
    print(data_TW)
    print()
    print("-------------------------------------------------------------------------------------------------------------")
    print()
    print("Feature Latent semantics\n")
    print(feature_TW)