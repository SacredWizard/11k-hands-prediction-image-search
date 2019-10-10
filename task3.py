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
from utils.excel import ExcelReader


def main():
    """Main function for the Task3 """
    excel_reader = ExcelReader()
    excel_reader.save_hand_csv_mongo("HandInfo.csv")


if __name__ == "__main__":
    main()
