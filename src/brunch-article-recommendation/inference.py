import argparse

import numpy as np
from tqdm import tqdm

from data.dataset import load_data, read_rowwise, valid, read_total, chainer, metadata
from model.collaborative import als_model
from model.content import content_model


def collective_inference():
    user2idx, idx2article, purchase_sparse = load_data(read_rowwise)
    popular_rec_model = read_rowwise["article_id"].value_counts().index[0:1000]
    rec_model = als_model(purchase_sparse)

    with open(args.path + args.file, "w") as f:
        for user in tqdm(valid[0].values):
            try:
                recs = rec_model.recommend(user2idx[user], purchase_sparse, N=150)
                recs = [idx2article[x[0]] for x in recs][0:100]
                f.write(f"{user} {' '.join(recs)}\n")
            except KeyError:
                recs = popular_rec_model[0:100]
                f.write(f"{user} {' '.join(recs)}\n")


def content_inference():
    idx2article, article2idx, articleidx, cos_sim = content_model(metadata)
    top_n = 100
    with open(args.path + args.file, "w") as f:
        for user in tqdm(valid[0].values):
            seen = chainer(read_total[read_total["user_id"] == user]["article_list"])
            for seen_id in seen:
                # 2019년도 이전에 읽어서 혹은 메타데이터에 글이 없어서 유사도 계산이 안된 글
                cos_sim_sum = np.zeros(len(cos_sim))
                try:
                    cos_sim_sum += cos_sim[article2idx[seen_id]]
                except KeyError:
                    pass
            recs = []
            for rec in cos_sim_sum.argsort()[-(top_n + 100) :][::-1]:
                if (idx2article[rec] not in seen) & (len(recs) < 100):
                    recs.append(idx2article[rec])
            f.write(f"{user} {' '.join(recs[0:100])}\n")


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Inference")
    parse.add_argument(
        "--path", type=str, help="Input recommend txt", default="../../res/"
    )
    parse.add_argument(
        "--file", type=str, help="Input file name", default="baseline.txt"
    )
    args = parse.parse_args()
    content_inference()
