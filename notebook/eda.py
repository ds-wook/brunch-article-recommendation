# %%
from typing import List
from itertools import chain
from datetime import datetime
from math import log
import os
import sys
import gc
import pickle

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# %%

path = "../input/"
print(os.listdir(path))

# %%

magazine = pd.read_json(path + "magazine.json", lines=True)
metadata = pd.read_json(path + "metadata.json", lines=True)
users = pd.read_json(path + "users.json", lines=True)

# %%

input_read_path = path + "read/"
# os.listdir : 해당 경로에 있는 모든 파일들을 불러오는 명령어
file_list = os.listdir(input_read_path)
print(file_list[0:2])

# %%

read_df_list = []
exclude_file_lst = [".2019010120_2019010121.un~"]
for file in tqdm(file_list):
    # 예외처리
    if file in exclude_file_lst:
        continue
    else:
        file_path = input_read_path + file
        df_temp = pd.read_csv(file_path, header=None, names=["raw"])
        # file명을 통해서 읽은 시간을 추출(from, to)
        df_temp["from"] = file.split("_")[0]
        df_temp["to"] = file.split("_")[1]
        read_df_list.append(df_temp)

read_df = pd.concat(read_df_list)
read_df.head()
# %%

read_df["user_id"] = read_df["raw"].apply(lambda x: x.split()[0])
read_df["article_id"] = read_df["raw"].apply(lambda x: x.split()[1:])
read_df.head()

# %%


def chainer(s: pd.Series) -> List[str]:
    return list(chain.from_iterable(s))


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
read_rowwise.head()
# %%

users.head()
# %%
print("사용자의 수: ", users.shape[0])
print("작가의 수: ", users[users["keyword_list"].apply(lambda x: len(x)) != 0].shape[0])

# %%
print(
    "구독하는 작가가 있는 사용자의 수: ",
    users[users["following_list"].apply(lambda x: len(x)) != 0].shape[0],
)

# %%
users[users["keyword_list"].apply(lambda x: len(x)) != 0].head(1)[
    "keyword_list"
].values[0][0:10]
# %%
users["following_count"] = users["following_list"].apply(lambda x: len(x))

following_cnt_by_id = pd.DataFrame(
    users.groupby("following_count")["id"].count()
).reset_index()

plt.figure(figsize=(20, 12))
sns.pointplot(x="following_count", y="id", data=following_cnt_by_id)
plt.xticks(rotation=60)
plt.show()

# %%

users["following_count"].describe()


# %%

following_cnt_by_user = users["following_list"].map(len)
following_rowwise = pd.DataFrame(
    {
        "user_id": np.repeat(users["id"], following_cnt_by_user),
        "author_id": chainer(users["following_list"]),
    }
)

following_rowwise.reset_index(drop=True, inplace=True)
following_rowwise.head()
# %%
following_cnt_by_id = (
    following_rowwise.groupby("author_id")["user_id"]
    .agg({"count"})
    .reset_index()
    .sort_values(by="count", ascending=False)
)
following_cnt_by_id.head(10)
# %%
plt.figure(figsize=(12, 8))
sns.distplot(following_cnt_by_id["count"], kde=False, bins=1000)
plt.xticks(rotation=60)
plt.show()
# %%
following_cnt_by_id["count"].describe()
# %%
keyword_dict = {}

for i in tqdm(
    users[users["keyword_list"].apply(lambda x: len(x)) != 0]["keyword_list"].values
):
    for j in range(len(i)):
        word = i[j]["keyword"]
        cnt = i[j]["cnt"]
        try:
            keyword_dict[word] += cnt
        except:
            keyword_dict[word] = cnt
keyword_dict
# %%

read_rowwise.head()
# %%

read_rowwise = read_rowwise[read_rowwise["article_id"] != ""].reset_index(drop=True)

# %%
read_rowwise["dt"] = (
    read_rowwise["from"].astype(str).apply(lambda x: x[0:8]).astype(int)
)
read_rowwise["hr"] = (
    read_rowwise["from"].astype(str).apply(lambda x: x[8:10]).astype(int)
)
read_rowwise["read_dt"] = pd.to_datetime(
    read_rowwise["dt"].astype(str).apply(lambda x: x[0:4] + "-" + x[4:6] + "-" + x[6:8])
)
read_rowwise.head()
# %%
read_rowwise["article_id"].value_counts()[0:5]
# %%
read_rowwise["author_id"] = read_rowwise["article_id"].apply(
    lambda x: str(x).split("_")[0]
)
read_rowwise["author_id"].value_counts()[:10]
# %%
following_rowwise["is_following"] = 1
read_rowwise = pd.merge(
    read_rowwise, following_rowwise, how="left", on=["user_id", "author_id"]
)
read_rowwise["is_following"] = read_rowwise["is_following"].fillna(0)
read_rowwise["is_following"].value_counts(normalize=True)

# %%

metadata.head()

# %%


metadata["reg_datetime"] = metadata["reg_ts"].apply(
    lambda x: datetime.fromtimestamp(x / 1000.0)
)
metadata.loc[
    metadata["reg_datetime"] == metadata["reg_datetime"].min(), "reg_datetime"
] = datetime(2090, 12, 31)
metadata["reg_dt"] = metadata["reg_datetime"].dt.date
metadata["type"] = metadata["magazine_id"].apply(lambda x: "개인" if x == 0.0 else "매거진")
metadata["reg_dt"] = pd.to_datetime(metadata["reg_dt"])


# %%

read_cnt_by_reg_dt = pd.DataFrame(
    metadata.groupby("reg_dt")["article_id"].count()
).reset_index()
read_cnt_by_reg_dt = read_cnt_by_reg_dt.iloc[:-1]
plt.figure(figsize=(25, 10))
sns.lineplot(x="reg_dt", y="article_id", data=read_cnt_by_reg_dt)
plt.xticks(rotation=60)
plt.show()

# %%
read_cnt_by_reg_dt_ = read_cnt_by_reg_dt[read_cnt_by_reg_dt["reg_dt"] >= "2019-03-01"]

plt.figure(figsize=(25, 10))
sns.lineplot(x="reg_dt", y="article_id", data=read_cnt_by_reg_dt_)
plt.xticks(rotation=60)
plt.show()

# %%

read_cnt_by_reg_dt = (
    read_rowwise[read_rowwise["author_id"] == "@basenell"]
    .groupby(["read_dt", "article_id"])["article_id"]
    .agg({"count"})
    .reset_index()
)
metadata_ = metadata[["id", "reg_dt"]].rename(columns={"id": "article_id"})
read_cnt_by_reg_dt = pd.merge(
    read_cnt_by_reg_dt, metadata_[["article_id", "reg_dt"]], how="left", on="article_id"
)
read_cnt_by_reg_dt = read_cnt_by_reg_dt[read_cnt_by_reg_dt["reg_dt"] >= "2019-01-15"]
plt.figure(figsize=(25, 10))
sns.lineplot(x="reg_dt", y="count", hue="article_id", data=read_cnt_by_reg_dt)
plt.xticks(rotation=60)
plt.show()

# %%
read_rowwise = pd.merge(read_rowwise, metadata_, how="left", on="article_id")
read_rowwise["diff_dt"] = (read_rowwise["read_dt"] - read_rowwise["reg_dt"]).dt.days
off_day = read_rowwise.groupby(["diff_dt"])["diff_dt"].agg({"count"}).reset_index()

off_day = off_day[(off_day["diff_dt"] >= 0) & (off_day["diff_dt"] <= 200)]
# %%
plt.figure(figsize=(25, 10))
sns.lineplot(x="diff_dt", y="count", data=off_day)
plt.xticks(rotation=60)
plt.show()

# %%
