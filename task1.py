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
from utils.termweight import print_tw
from utils.model import Model
from utils.excelcsv import CSVReader
from utils.imageviewer import show_feature_ls
from classes.globalconstants import GlobalConstants
from utils.inputhelper import get_input_dimensionality_reduction_model, get_input_feature_extractor_model, get_input_k

model_interact = Model()
global_constants = GlobalConstants()


def save_model(dim_reduction, feature_extraction_model, dimension_reduction_model, k_value):
    obj_lat, feat_lat, model = dim_reduction.execute()
    # Saves the returned model
    filename = feature_extraction_model + "_" + dimension_reduction_model + "_" + str(k_value)
    model_interact.save_model(model=model, filename=filename)
    return obj_lat, feat_lat


def main():
    """Main function for the task 1"""
    feature_extraction_model = get_input_feature_extractor_model()
    dimension_reduction_model = get_input_dimensionality_reduction_model()
    k_value = get_input_k()

    print(global_constants.LINE_SEPARATOR)
    print("User Inputs summary")
    print(global_constants.LINE_SEPARATOR)
    print("\nFeature Extraction Model: {}\nDimensionality Reduction Model: {}\nk-value: {}\n".
          format(feature_extraction_model, dimension_reduction_model, k_value))
    print(global_constants.LINE_SEPARATOR)

    # Performs the dimensionality reduction
    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value)
    obj_lat, feat_lat = save_model(dim_reduction, feature_extraction_model, dimension_reduction_model, k_value)

    # Print term weight pairs to terminal  
    print_tw(obj_lat, feat_lat)

    # save term weight pairs to csv  
    filename = "task1" + '_' + feature_extraction_model + '_' + dimension_reduction_model + '_' + str(k_value)
    CSVReader().save_to_csv(obj_lat, feat_lat, filename)
    print("Please check the CSV file: output/{}.csv".format(filename))

    data = dim_reduction.get_object_feature_matrix()

    title = {
        "Feature Extraction": feature_extraction_model,
        "Dimensionality Reduction": dimension_reduction_model,
        "k": k_value,
    }
    show_feature_ls(data, feat_lat, title)


if __name__ == "__main__":
    main()
