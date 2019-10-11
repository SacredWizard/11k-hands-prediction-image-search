"""
Multimedia Web Databases - Fall 2019: Project Group 17
Authors:
1. Sumukh Ashwin Kamath
2. Rakesh Ramesh
3. Baani Khurana
4. Karishma Joseph
5. Shantanu Gupta
6. Kanishk Bashyam

This is the CLI for task 2 of Phase 2 of the project
"""
from classes.dimensionreduction import DimensionReduction
from utils.model import Model
from classes.featureextraction import ExtractFeatures

model_interact = Model()


def main():
    """Main function for the task 2"""
    feature_extraction_model = "HOG"
    dimension_reduction_model = "LDA"
    folder = "testset2/"
    image_name = "Hand_0009885.jpg"
    dist_func = "euclidean"
    k_value = 10
    m_value = 5

    # Saves the returned model
    filename = feature_extraction_model + "_" + dimension_reduction_model + "_" + str(k_value)
    model = model_interact.load_model(filename=filename)

    # Compute the reduced dimensions for the new query image
    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value)
    result = dim_reduction.find_m_similar_images(model, m_value, folder, image_name, dist_func)
    for rec in result:
        print(rec)


if __name__ == "__main__":
    main()
