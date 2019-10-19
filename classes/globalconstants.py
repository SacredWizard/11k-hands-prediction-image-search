class GlobalConstants:

    def __init__(self):
        self.JPG_EXTENSION = '.jpg'
        self.HOG = 'HOG'
        self.CM = 'CM'
        self.SIFT = 'SIFT'
        self.LBP = 'LBP'
        self.PCA = 'PCA'
        self.SVD = 'SVD'
        self.NMF = 'NMF'
        self.LDA = 'LDA'
        self.FEATURE_MODELS = [self.CM, self.HOG, self.LBP, self.SIFT]
        self.REDUCTION_MODELS = [self.PCA, self.SVD, self.NMF, self.LDA]
        self.image_labels = ["dorsal", "palmar", "left-hand", "right-hand", "with accessories", "without accessories",
                             "male", "female"]
        self.METADATA = "metadata"
        self.WINDOW_SIZE = 100
        self.BULK_PROCESS_COUNT = 50
        self.NUM_THREADS = 4
        self.MODELS_FOLDER = 'models'
        self.BOW_MODEL = 'KNN'
        self.BOW_BATCH_SIZE = 50

    class Mongo:

        def __init__(self):
            self.MONGO_URL = 'mongodb://localhost:27017'
            self.MONGO_SERVER_TIMEOUT = 2000
            self.BULK_INSERT_REC_COUNT = 50
            self.DB_NAME = 'mwdb'
            self.SIFT_FEATURE_COLLECTION = 'sift_features'
            self.METADATA_DB_NAME = "metadata"

    class Lbp:

        def __init__(self):
            self.BINS = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96,
                         112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223,
                         224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255, 256]
            self.NUM_POINTS = 8
            self.RADIUS = 1
            self.METHOD_UNIFORM = 'uniform'

    class Nmf:

        def __init__(self):
            self.BETA_LOSS_FROB = 'frobenius'
            self.INIT_MATRIX = 'nndsvd'