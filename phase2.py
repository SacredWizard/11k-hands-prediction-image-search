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
from classes.dimensionreduction import DimensionReduction


def main():
    """Main function for Phase 2"""
    dim_reduce = DimensionReduction("HOG", "NMF")
    dim_reduce.execute()


if __name__ == "__main__":
    main()
