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
csv_reader = CSVReader()


def main():
    """Main function for the script"""
    input_data = {"labelled": ["Dataset3/labelled_set1.csv", "Dataset3/labelled_set2.csv"],
                  "unlabelled": ["Dataset3/unlabelled_set1.csv", "Dataset3/unlabelled_set2.csv"]}
    csv_reader.save_csv_multiple(input_data)


if __name__ == "__main__":
    main()
