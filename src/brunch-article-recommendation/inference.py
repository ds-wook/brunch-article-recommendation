import argparse

from tqdm import tqdm

from data.dataset import load_data, read_rowwise, valid
from model.collaborative import als_model


def __main():
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


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Inference")
    parse.add_argument(
        "--path", type=str, help="Input recommend txt", default="../../res/"
    )
    parse.add_argument(
        "--file", type=str, help="Input file name", default="baseline.txt"
    )
    args = parse.parse_args()
    __main()
