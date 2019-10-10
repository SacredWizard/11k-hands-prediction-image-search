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


def main():
    """Main function for the task 1"""
    feature_extraction_model = "HOG"
    dimension_reduction_model = "LDA"
    k_value = 10
    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value)
    dim_reduction.execute()


if __name__ == "__main__":
    main()
