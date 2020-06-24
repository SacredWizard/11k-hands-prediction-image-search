# 11k Hands, Prediction and Image Search


The dependencies for this project are present in `environment.yml`

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

Used 11k hand images dataset. Extracted the features using Color Moments (CM), Histogram of Oriented Gradients (HOG), Local Binary Pattens (LBP), Scale Invariant Feature Transforms (SIFT). Implemented Dimensionality reduction using Principal Component Analysis (PCA), Singular Vector Decomposition (SVD), Nonnegative Matrix Factorization (NMF) and Latent Dirichlet Analysis (LDA). Finds similar subjects using an input hand image. Implemented Subject-Subject similarity. Identifies the gender of unlabelled images. Implemented image similarity graph using Personalized Page Rank (PPR). Implemented Support Vector Machine (SVM), Decision Tree Classifier and Personalized page Rank Classifier. Implemented Locality Sensitive Hashing (LSH) and used it to find similar images in the dataset and created a visual model. Implemented Relevance Feedback System using SVM, decision tree, PPR based relevance feedback system and Probabilistic Relevance feedback system and categorized relevant and irrelevant images. 
