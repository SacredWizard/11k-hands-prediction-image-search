from numpy import argsort

def get_data_latent_semantics(data_m, k):
    term = [argsort(-data_m[:, i]) for i in range(k)]
    tw=[]
    #for each latent semantic (k)
    for i in range(k):
        k_tw = []
        #for each data object
        for j in range(len(term[i])):
            k_tw.append([term[i][j],data_m[term[i][j],i]])
        tw.append(k_tw)
    return tw

def get_feature_latent_semantics(feature_m, k):
    term = [argsort(-feature_m[i, :]) for i in range(k)]
    tw=[]
    #for each latent semantic (k)
    for i in range(k):
        k_tw = []
        #for each feature
        for j in range(len(term[i])):
            k_tw.append([term[i][j],feature_m[i,term[i][j]]])
        tw.append(k_tw)
    return tw

    #prints Term Weight Pairs to console
def print_tw(data_tw, feature_tw, feature_descriptor):
    print()
    print("-------------------------------------------------------------------------------------------------------------")
    print()
    print("Data Latent Semantics\n")
    print(data_tw)
    print()
    print("-------------------------------------------------------------------------------------------------------------")
    print()
    print("Feature Latent Semantics\n")
    print(feature_tw)