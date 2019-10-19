from utils.inputhelper import get_task_number
import importlib


def main():
    """Main function for the script"""
    number_of_tasks = 8
    choice = get_task_number(8)
    module_name = "task{}".format(choice)
    module = importlib.import_module(module_name)
    module.main()


if __name__ == "__main__":
    main()
