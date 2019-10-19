"""
Multimedia Web Databases - Fall 2019: Project Group 17
Authors:
1. Sumukh Ashwin Kamath
2. Rakesh Ramesh
3. Baani Khurana
4. Karishma Joseph
5. Shantanu Gupta
6. Kanishk Bashyam
This is the CLI for task 5 of Phase 2 of the project
"""
from classes.dimensionreduction import DimensionReduction
from utils.model import Model
from utils.excelcsv import CSVReader

model_interact = Model()


def run_task3(feature_extraction_model, dimension_reduction_model, folder, dist_func, label, k_value):
    """Main function for the Task3"""
    # Performs the dimensionality reduction
    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value, label)
    obj_feature = dim_reduction.get_object_feature_matrix()
    obj_lat, feat_lat, model = dim_reduction.execute()

    # Saves the returned model
    filename = "{0}_{1}_{2}_{3}".format(feature_extraction_model, dimension_reduction_model, label.replace(" ", ''),
                                        str(k_value))
    model_interact.save_model(model=model, filename=filename)


def run_task4(feature_extraction_model, dimension_reduction_model, folder, image_name, dist_func, label, k_value, m_value):
    """Main function for the Task4"""

    # Saves the returned model
    filename = "{0}_{1}_{2}_{3}".format(feature_extraction_model, dimension_reduction_model, label.replace(" ", ''),
                                        str(k_value))
    model = model_interact.load_model(filename=filename)

    # Compute the reduced dimensions for the new query image and find m similar images
    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value, label)
    obj_feature = dim_reduction.get_object_feature_matrix()
    result = dim_reduction.find_m_similar_images(model, m_value, folder, image_name, dist_func)
    return result


def get_class_labels(label):
    class1_labels = ["left-hand", "dorsal", "with accessories", "male"]
    class2_labels = ["right-hand", "palmar", "without accessories", "female"]
    if label in class1_labels:
        label_type = 1
        index = class1_labels.index(label)
    else:
        label_type = 2
        index = class2_labels.index(label)
    if label_type == 1:
        class1_label = label
        class2_label = class2_labels[index]
    else:
        class2_label = label
        class1_label = class1_labels[index]
    
    return class1_label, class2_label


def main():
    feature_extraction_model = "CM"
    dimension_reduction_model = "SVD"
    folder = "testset4/"
    query_folder = "testset4/"
    image_name = "Hand_0003521.jpg"
    dist_func = "euclidean"
    k_value = 10
    m_value = 1
    label = "male"

    class1_label, class2_label = get_class_labels(label)

    run_task3(feature_extraction_model, dimension_reduction_model, folder, dist_func, class1_label, k_value)
    result1 = run_task4(feature_extraction_model, dimension_reduction_model, query_folder, image_name, dist_func, class1_label, k_value, m_value)
    # for rec in result1:
    #     print(rec['score'])
    class1_score = result1[0]['score']
    # print(class1_score)

    run_task3(feature_extraction_model, dimension_reduction_model, folder, dist_func, class2_label, k_value)
    result2 = run_task4(feature_extraction_model, dimension_reduction_model, query_folder, image_name, dist_func, class2_label, k_value, m_value)
    # for rec in result2:
    #     print(rec['score'])
    class2_score = result2[0]['score']
    # print(class2_score)

    final_label = class1_label if class1_score > class2_score else class2_label

    print(final_label)


if __name__ == "__main__":
    main()
