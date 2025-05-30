import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

class KNNContradictionDetector:
    def __init__(self, k=3):
        self.k = k
        self.knn = None
        self.embeddings = None
        self.labels = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the KNN model using embeddings and contradiction labels.
        """
        self.embeddings = X
        self.labels = y
        self.knn = KNeighborsClassifier(n_neighbors=self.k, metric='cosine')
        self.knn.fit(X, y)

    def predict(self, x_new: np.ndarray) -> int:
        """
        Predict if a new policy intent is a contradiction (1) or not (0).
        """
        return self.knn.predict([x_new])[0]

    def contradiction_score(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute cosine similarity as a contradiction score.
        Lower similarity = higher chance of contradiction.
        """
        sim = cosine_similarity([x1], [x2])[0][0]
        return 1 - sim  # contradiction score
