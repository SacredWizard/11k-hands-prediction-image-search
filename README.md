# CSE515GroupProject
MWDB Group 17 Project

Clone the repo and Start coding !!

The dependencies for this project are present in `environment.yml`



* To create an environment, execute
    ```shell script
    conda env create -f config/environment.yml 
    ```
* When dependencies are added to `environment.yml` execute
    ```shell script
    conda deactivate
    conda env update -f config/environment.yml --prune
    conda activate CSE515
    ```
* To remove an environment
    ```shell script
    conda remove --name <env_name> --all
    ```