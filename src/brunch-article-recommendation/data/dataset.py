import os
import itertools
from itertools import chain
from typing import List, Tuple, Dict
from datetime import datetime

import pandas as pd
import numpy as np
import scipy
from scipy.sparse import csc_matrix

from tqdm import tqdm


def chainer(s: pd.Series) -> List[str]:
    return list(chain.from_iterable(s))


path = "../../input/"
magazine = pd.read_json(path + "magazine.json", lines=True)
metadata = pd.read_json(path + "metadata.json", lines=True)
users = pd.read_json(path + "users.json", lines=True)

input_read_path = path + "read/"

read_df_list = []
exclude_file_lst = [".2019010120_2019010121.un~"]
file_list = os.listdir(input_read_path)

for f in tqdm(file_list):
    # 예외처리
    if f in exclude_file_lst:
        continue
    else:
        file_path = input_read_path + f
        df_temp = pd.read_csv(file_path, header=None, names=["raw"])
        # file명을 통해서 읽은 시간을 추출(from, to)
        df_temp["from"] = f.split("_")[0]
        df_temp["to"] = f.split("_")[1]
        read_df_list.append(df_temp)

read_df = pd.concat(read_df_list)
read_df["user_id"] = read_df["raw"].apply(lambda x: x.split()[0])
read_df["article_id"] = read_df["raw"].apply(lambda x: x.split()[1:])

read_cnt_by_user = read_df["article_id"].map(len)
read_rowwise = pd.DataFrame(
    {
        "from": np.repeat(read_df["from"], read_cnt_by_user),
        "to": np.repeat(read_df["to"], read_cnt_by_user),
        "user_id": np.repeat(read_df["user_id"], read_cnt_by_user),
        "article_id": chainer(read_df["article_id"]),
    }
)
read_rowwise.reset_index(drop=True, inplace=True)

metadata["reg_datetime"] = metadata["reg_ts"].apply(
    lambda x: datetime.fromtimestamp(x / 1000.0)
)
metadata.loc[
    metadata["reg_datetime"] == metadata["reg_datetime"].min(), "reg_datetime"
] = datetime(2090, 12, 31)
metadata["reg_dt"] = metadata["reg_datetime"].dt.date
metadata["type"] = metadata["magazine_id"].apply(lambda x: "개인" if x == 0.0 else "매거진")
metadata["reg_dt"] = pd.to_datetime(metadata["reg_dt"])


valid = pd.read_csv(path + "/predict/dev.users", header=None)


def load_data(
    read_rowwise: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[int, str], csc_matrix]:
    # 협업 필터링 기반의 모델
    read_rowwise = read_rowwise.merge(
        metadata[["id", "reg_dt"]], how="left", left_on="article_id", right_on="id"
    )
    read_rowwise = read_rowwise[read_rowwise["article_id"] != ""]

    read_total = pd.DataFrame(
        read_rowwise.groupby(["user_id"])["article_id"].unique()
    ).reset_index()
    read_total.columns = ["user_id", "article_list"]

    read_rowwise = read_rowwise[
        (read_rowwise["id"].notnull()) & (read_rowwise["reg_dt"].notnull())
    ]
    read_rowwise = read_rowwise[
        (read_rowwise["reg_dt"] >= "2019-01-01")
        & (read_rowwise["reg_dt"] < "2090-12-31")
    ].reset_index(drop=True)

    del read_rowwise["id"]

    # read_total_valid = read_total[read_total["user_id"].isin(valid[0])].reset_index(
    #     drop=True
    # )
    read_total_train = read_total[~read_total["user_id"].isin(valid[0])].reset_index(
        drop=True
    )

    read_total_train["article_len"] = read_total_train["article_list"].apply(
        lambda x: len(x)
    )
    top10_percent = np.percentile(read_total_train["article_len"].values, 90)
    read_total_train = read_total_train[
        read_total_train["article_len"] >= top10_percent
    ]
    hot_user = read_total_train["user_id"].unique()

    user_total = pd.DataFrame(
        read_rowwise.groupby(["article_id"])["user_id"].unique()
    ).reset_index()
    user_total.columns = ["article_id", "user_list"]

    user_total["user_len"] = user_total["user_list"].apply(lambda x: len(x))
    cold_article = user_total[user_total["user_len"] <= 20]["article_id"].unique()

    read_rowwise = read_rowwise[
        read_rowwise["user_id"].isin(np.append(hot_user, valid[0].values))
    ]
    read_rowwise = read_rowwise[~read_rowwise["article_id"].isin(cold_article)]

    user2idx = {l: i for i, l in enumerate(read_rowwise["user_id"].unique())}
    article2idx = {l: i for i, l in enumerate(read_rowwise["article_id"].unique())}
    # idx2user = {i: user for user, i in user2idx.items()}
    idx2article = {i: item for item, i in article2idx.items()}

    data = read_rowwise[["user_id", "article_id"]].reset_index(drop=True)
    useridx = data["userid"] = (
        read_rowwise["user_id"].apply(lambda x: user2idx[x]).values
    )
    articleidx = data["articleidx"] = (
        read_rowwise["article_id"].apply(lambda x: article2idx[x]).values
    )
    rating = np.ones(len(data))
    purchase_sparse = scipy.sparse.csr_matrix(
        (rating, (useridx, articleidx)), shape=(len(set(useridx)), len(set(articleidx)))
    )

    return user2idx, idx2article, purchase_sparse
