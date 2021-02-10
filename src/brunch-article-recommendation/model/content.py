from typing import Dict, Tuple
import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def content_model(
    metadata: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[int, str], Dict[int, str], cosine_similarity]:
    metadata = metadata[metadata["keyword_list"].notnull()].reset_index()
    metadata = metadata[metadata["reg_dt"] >= "2019-01-01"]
    article2idx = {l: i for i, l in enumerate(metadata["id"].unique())}
    idx2article = {i: item for item, i in article2idx.items()}
    articleidx = metadata["articleidx"] = (
        metadata["id"].apply(lambda x: article2idx[x]).values
    )
    docs = metadata["keyword_list"].apply(lambda x: " ".join(x)).values
    tfidv = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None).fit(docs)
    tfidv_df = scipy.sparse.csr_matrix(tfidv.transform(docs))
    tfidv_df = tfidv_df.astype(np.float32)
    cos_sim = cosine_similarity(tfidv_df, tfidv_df)

    return idx2article, article2idx, articleidx, cos_sim
