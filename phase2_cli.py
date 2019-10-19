"""
Measuring Similarity of Hand Images Using Color Moments and Histogram of Oriented Gradients
This is the CLI for the Second Phase of the Project
Author : Sumukh Ashwin Kamath
(ASU ID - 1217728013 email - skamath6@asu.edu
"""
from utils.inputhelper import get_task_number
import importlib


def main():
    """Main function for the script"""
    number_of_tasks = 8
    choice = get_task_number(number_of_tasks)
    module_name = "task{}".format(choice)
    module = importlib.import_module(module_name)
    module.main()


if __name__ == "__main__":
    main()
