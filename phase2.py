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
import utils.termweight as tw
from utils.excelcsv import CSVReader
from classes.dimensionreduction import DimensionReduction
import numpy as np

def main():
    """Main function for Phase 2"""

    feature_descriptor = "LBP"
    dr_method = "NMF"
    k = 5

    dim_reduce = DimensionReduction(feature_descriptor, dr_method, k)
    data_m, feature_m, model = dim_reduce.execute()

    # prints all term weight pairs
    tw.print_tw(data_m, feature_m)

    # save term weight pairs to csv  
    excsv = CSVReader()
    excsv.save_to_csv(data_tw, feature_tw)

if __name__ == "__main__":
    main()
