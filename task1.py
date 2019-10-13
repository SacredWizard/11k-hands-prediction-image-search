"""
Multimedia Web Databases - Fall 2019: Project Group 17
Authors:
1. Sumukh Ashwin Kamath
2. Rakesh Ramesh
3. Baani Khurana
4. Karishma Joseph
5. Shantanu Gupta
6. Kanishk Bashyam

This is the CLI for task 1 of Phase 2 of the project
"""
from classes.dimensionreduction import DimensionReduction
from utils.model import Model

model_interact = Model()


def main():
    """Main function for the task 1"""
    feature_extraction_model = "HOG"
    dimension_reduction_model = "NMF"
    k_value = 10

    # Performs the dimensionality reduction
    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value)
    obj_lat, feat_lat, model = dim_reduction.execute()

    # Saves the returned model
    filename = feature_extraction_model + "_" + dimension_reduction_model + "_" + str(k_value)
    model_interact.save_model(model=model, filename=filename)


if __name__ == "__main__":
    main()
