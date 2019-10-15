"""
Multimedia Web Databases - Fall 2019: Project Group 17
Authors:
1. Sumukh Ashwin Kamath
2. Rakesh Ramesh
3. Baani Khurana
4. Karishma Joseph
5. Shantanu Gupta
6. Kanishk Bashyam

This is the CLI for task 4 of Phase 2 of the project
"""
import os
from classes.dimensionreduction import DimensionReduction
from utils.imageviewer import show_images
from utils.model import Model

model_interact = Model()


def main():
    """Main function for the task 4"""
    feature_extraction_model = "HOG"
    dimension_reduction_model = "LDA"
    folder = "testset2/"
    image_name = "Hand_0003496.jpg"
    dist_func = "euclidean"
    label = "with accessories"
    k_value = 10
    m_value = 5

    # Saves the returned model
    filename = "{0}_{1}_{2}_{3}".format(feature_extraction_model, dimension_reduction_model, label.replace(" ", ''),
                                        str(k_value))
    model = model_interact.load_model(filename=filename)

    # Compute the reduced dimensions for the new query image and find m similar images
    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value, label)
    result = dim_reduction.find_m_similar_images(model, m_value, folder, image_name, dist_func)
    for rec in result:
        print(rec)
    show_images(os.path.abspath(os.path.join(folder, image_name)), result)


if __name__ == "__main__":
    main()
