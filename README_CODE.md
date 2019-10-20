# README

This project requires MongoDB. We will also need to install OpenCV and OpenCV-Contrib to use the SIFT algorithm. We will also require python 3.7.4 and numpy.

## Installation



1. Follow the instructions from the MongoDB Manual to install MongoDB and start the mongoDB server
   `https://docs.mongodb.com/manual/installation/`
2. Install OpenCV with Qt4 and OpenCV-Contrib 4.1.1
   use the following guide:
   [Medium guide to install openCV and openCV-Contrib on MacOS](https://medium.com/repro-repo/install-opencv-4-0-1-from-source-on-macos-with-anaconda-python-3-7-to-use-sift-and-surf-9d4287d6228b)
   Follow this guide till step 3. If you do not wish to use the guide proceed to step 4 below.
    
3. Build and install the package using the following cmake command:
     ```sh
     cmake -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX=$cwd/installation/OpenCV-"$cvVersion" \
            -D INSTALL_C_EXAMPLES=ON \
            -D INSTALL_PYTHON_EXAMPLES=ON \
            -D WITH_TBB=ON \
            -D WITH_V4L=ON \
            -D OPENCV_SKIP_PYTHON_LOADER=ON \
            -D CMAKE_PREFIX_PATH=$QT5PATH \
            -D CMAKE_MODULE_PATH="$QT5PATH"/lib/cmake \
            -D OPENCV_PYTHON3_INSTALL_PATH=~/.virtualenvs/OpenCV-"$cvVersion"-py3/lib/python3.7/site-packages \
        -D WITH_QT=ON \
        -D WITH_OPENGL=ON \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D BUILD_EXAMPLES=ON ..
   ``` 
   and then `make -j$(sysctl -n hw.physicalcpu)` `make install`
4. To install using the dependencies list, just use the `config/environment.yml` it gives the list of all the packages used.
    * To create an environment, execute
        ```shell script
        conda env create -f config/environment.yml 
        ```
    * When dependencies are added to `environment.yml` execute
        ```shell script
        conda deactivate
        conda env update -f config/environment.yml --prune
        conda activate <env_name>
        ```
    * To remove an environment
        ```shell script
        conda remove --name <env_name> --all
        ```
 

 



## Running the code
To Run the code, you must have MongoDB running, and then run the following command
```shell script
conda activate CSE515
python phase2_cli.py
```
