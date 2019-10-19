"""
Multimedia Web Databases - Fall 2019: Project Group 17
Authors:
1. Sumukh Ashwin Kamath
2. Rakesh Ramesh
3. Baani Khurana
4. Karishma Joseph
5. Shantanu Gupta
6. Kanishk Bashyam

This is the CLI for task 8 of Phase 2 of the project
"""

from utils.inputhelper import get_input_k, get_input_folder
from classes.dimensionreduction import DimensionReduction
from utils.termweight import print_tw
from utils.excelcsv import CSVReader


def main():
    """Main function for the Task 8"""
    k_value = get_input_k()
    folder = get_input_folder()
    dim_red = DimensionReduction(None, "NMF", k_value, image_metadata=True, folder_metadata=folder)
    w, h, model = dim_red.execute()

    # printing the term weight
    print_tw(w, h, image_metadata=True)

    # save to csv
    filename = "task8" + "_" + str(k_value)
    CSVReader().save_to_csv(w, h, filename, image_metadata=True)


if __name__ == "__main__":
    main()
