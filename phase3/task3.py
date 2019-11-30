"""
Multimedia Web Databases - Fall 2019: Project Group 17
Authors:
1. Sumukh Ashwin Kamath
2. Rakesh Ramesh
3. Baani Khurana
4. Karishma Joseph
5. Shantanu Gupta
6. Kanishk Bashyam

This is the CLI for loading the metadata on to mongo
"""
from classes.dimensionreduction import DimensionReduction
import time


def main():
    """Main function for the script"""
    start = time.time()
    feature_extraction_model = "HOG"
    dimension_reduction_model = "PCA"
    k_value = 40
    folder = "Dataset3/Labelled/Set1"
    dim_red = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value, folder_metadata=folder,
                                 metadata_collection="labelled")
    print(dim_red.get_object_feature_matrix()[['featureVector', 'imageId']])
    print("Execution time: {} seconds".format(time.time() - start))


if __name__ == "__main__":
    main()
