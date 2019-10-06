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
from classes.dimensionreduction import DimensionReduction


def main():
    """Main function for Phase 2"""

    feature_descriptor = "CM"
    dr_method = "SVD"
    k = 3

    dim_reduce = DimensionReduction(feature_descriptor, dr_method, k)
    data_m, feature_m = dim_reduce.execute()

    # gets data term weight pairs
    data_tw = tw.get_data_latent_semantics(data_m, k)

    # gets feature term weight pairs
    feature_tw = tw.get_feature_latent_semantics(feature_m, k)

    # prints all term weight pairs
    tw.print_tw(data_tw, feature_tw, feature_descriptor)


if __name__ == "__main__":
    main()
