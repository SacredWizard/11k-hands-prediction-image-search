import os

from classes.globalconstants import GlobalConstants

constants = GlobalConstants()


def validate_image(folder, image):
    return image.endswith(constants.JPG_EXTENSION) and os.path.isfile(os.path.join(folder, image))


def validate_folder(folder):
    return os.path.isdir(folder)


def validate_images(folder):
    if os.path.isdir(folder):
        pass
