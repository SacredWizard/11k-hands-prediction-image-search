"""
Image Search - Multimedia Web Database Systems Fall 2019 Project Group 17
This is the file for interacting with CSV Files
Author : Sumukh Ashwin Kamath
(ASU ID - 1217728013 email - skamath6@asu.edu
"""
import os
import matplotlib.pyplot as plt
from skimage import io


def show_images(image_list):
    plt.rcParams.update({'font.size': 10})
    """Visualizer for the images"""
    f = plt.figure()

    no_images_per_row = 5
    no_of_lines = int(len(image_list) / no_images_per_row + 1)
    print(no_of_lines)

    # count = 1
    # for data in data_tw[:2]:
    #     for r in data[:5]:
    #         f.add_subplot(no_of_lines, 5, count)
    #         plt.imshow(io.imread(os.path.abspath(os.path.join("Dataset2/", r[0]))))
    #         plt.title(
    #             "{}\nScore: {}".format(r[0], r[1]))
    #         plt.axis('off')
    #         count = count + 1
    #     plt.show()

    count = 1
    for r in image_list:
        f.add_subplot(no_of_lines, no_images_per_row, count)
        plt.imshow(io.imread(os.path.join(r['path'])))
        plt.title(
            "Latent semantic {} \n{}".format(count, r['imageId']))
        plt.axis('off')
        count = count + 1
    plt.show()


# import matplotlib
# # Make sure that we are using QT5
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
# from PyQt5 import QtWidgets
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# import os
# from skimage import io


# class ScrollableWindow(QtWidgets.QMainWindow):
#     def __init__(self, fig):
#         self.qapp = QtWidgets.QApplication([])

#         QtWidgets.QMainWindow.__init__(self)
#         self.widget = QtWidgets.QWidget()
#         self.setCentralWidget(self.widget)
#         self.widget.setLayout(QtWidgets.QVBoxLayout())
#         self.widget.layout().setContentsMargins(0,0,0,0)
#         self.widget.layout().setSpacing(10)

#         self.fig = fig
#         self.canvas = FigureCanvas(self.fig)
#         self.canvas.draw()
#         self.scroll = QtWidgets.QScrollArea(self.widget)
#         self.scroll.setWidget(self.canvas)

#         self.nav = NavigationToolbar(self.canvas, self.widget)
#         self.widget.layout().addWidget(self.nav)
#         self.widget.layout().addWidget(self.scroll)

#         self.show()
#         exit(self.qapp.exec_()) 

    

# #create a figure and some subplots
# fig, axes = plt.subplots(ncols=50, nrows=5, figsize=(16,16))
# for ax in axes.flatten():
#     ax.plot([2,3,5,1])

# #pass the figure to the custom window
# a = ScrollableWindow(fig)
