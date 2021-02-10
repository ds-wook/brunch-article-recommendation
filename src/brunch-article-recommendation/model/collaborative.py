import scipy
from implicit.als import AlternatingLeastSquares as ALS


def als_model(purchase_sparse: scipy.sparse.csc_matrix) -> ALS:
    als_model = ALS(factors=20, regularization=0.08, iterations=20)
    als_model.fit(purchase_sparse.T)
    return als_model
