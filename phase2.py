"""
Multimedia Web Databases - Fall 2019: Project Group 17
Authors:
1. Sumukh Ashwin Kamath
2. Rakesh Ramesh
3. Baani Khurana
4. Karishma Joseph
5. Shantanu Gupta
6. Kanishk Bashyam

This is a module for performing feature extraction on images
"""
from classes.dimensionreduction import DimensionReduction
import utils.termweight as tw

def main():
    """Main function for Phase 2"""
   
   
    feature_descriptor = "CM"
    dr_method = "SVD"
    k = 3

    dim_reduce = DimensionReduction(feature_descriptor, dr_method, k)
    data_M, feature_M = dim_reduce.execute()

    #gets data term weight pairs
    data_TW = tw.getDataLatentSemantics(data_M, k)

    #gets feature term weight pairs
    feature_TW = tw.getFeatureLatentSemantics(feature_M, k)

    #prints all term weight pairs
    tw.printTW(data_TW, feature_TW, feature_descriptor)

if __name__ == "__main__":
    main()
