"""
Multimedia Web Databases - Fall 2019: Project Group 17
Authors:
1. Sumukh Ashwin Kamath
2. Rakesh Ramesh
3. Baani Khurana
4. Karishma Joseph
5. Shantanu Gupta
6. Kanishk Bashyam

This is the CLI for Phase3 of the Project
"""
from utils.inputhelper import get_task_number
import importlib
import warnings

warnings.filterwarnings('ignore')


def main():
    """Main function for the script"""
    number_of_tasks = 6
    print("Welcome to Phase 3!")
    # choice = get_task_number(number_of_tasks)
    # choice = "load_csv"
    choice = 3
    if choice == "load_csv":
        module_name = "phase3.load_csv"
    elif choice == 4:
        module_name = "phase3.task4_ppr_1"
    elif choice == 6:
        module_name = "task6_svm"
    elif choice == 3:
        module_name = "phase3.task3"
    else:
        module_name = "task{}".format(choice)
    module = importlib.import_module('{0}'.format(module_name))
    module.main()


if __name__ == "__main__":
    main()


