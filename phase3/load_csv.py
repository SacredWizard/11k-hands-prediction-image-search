"""
Multimedia Web Databases - Fall 2019: Project Group 17
Authors:
1. Sumukh Ashwin Kamath
2. Rakesh Ramesh
3. Baani Khurana
4. Karishma Joseph
5. Shantanu Gupta
6. Kanishk Bashyam

This is the CLI for loading the metadata on to mongo
"""
from utils.excelcsv import CSVReader
import os
csv_reader = CSVReader()


def get_input(name):
    list_files = input("{}:".format(name))
    list_files = "".join(list_files.split(" ")).split(",")
    for file in list_files:
        if not os.path.isfile(file):
            print("Enter the correct filename")
            return get_input(name)

    return list_files


def main():
    """Main function for the script"""
    lab_path = get_input("Labelled CSV")
    unlab_path = get_input("Unlabelled CSV")
    input_data = {"labelled": lab_path,
                  "unlabelled": unlab_path,
                  "metadata": ["HandInfo.csv"]}
    csv_reader.save_csv_multiple(input_data)


if __name__ == "__main__":
    main()
