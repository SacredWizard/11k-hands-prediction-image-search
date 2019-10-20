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
from utils.inputhelper import get_input_feature_extractor_model, get_input_dimensionality_reduction_model, \
    get_input_folder, get_input_image, get_input_image_label, get_input_k, get_input_m

model_interact = Model()


def main():
    """Main function for the task 4"""
    feature_extraction_model = get_input_feature_extractor_model()
    dimension_reduction_model = get_input_dimensionality_reduction_model()
    k_value = get_input_k()
    label = get_input_image_label()
    folder = get_input_folder()
    image_name = get_input_image(folder)
    dist_func = "euclidean"
    m_value = get_input_m()

    # Saves the returned model
    filename = "{0}_{1}_{2}_{3}".format(feature_extraction_model, dimension_reduction_model, label.replace(" ", ''),
                                        str(k_value))
    model = model_interact.load_model(filename=filename)

    # Compute the reduced dimensions for the new query image and find m similar images
    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value, label)
    result = dim_reduction.find_m_similar_images(model, m_value, folder, image_name, dist_func)
    for rec in result:
        print(rec)

    title = {
        "Feature Extraction": feature_extraction_model,
        "Dimension Reduction": dimension_reduction_model,
        "k": k_value,
        "Label": label,
        "Distance": dist_func
             }
    show_images(os.path.abspath(os.path.join(folder, image_name)), result, title)


if __name__ == "__main__":
    main()
