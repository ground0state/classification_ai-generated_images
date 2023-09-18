# %%
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
# %%
file_list = [
    "work_dirs/exp09/submission_exp09_val_group_1.csv",
    "work_dirs/exp09_2/submission_exp09_2_val_group_2.csv",
    "work_dirs/exp09_3/submission_exp09_3_val_group_3.csv",
    "work_dirs/exp09_4/submission_exp09_4_val_group_4.csv",
    "work_dirs/exp17/submission_exp17_val_group_1_.csv",
    "work_dirs/exp18/submission_exp18_val_group_1_.csv"
]

df_list = []
for i, path in enumerate(file_list, start=1):
    df = pd.read_csv(path, header=None, index_col=0)
    df.index.name = "filename"
    df.columns = [f"score_{i}"]
    df_list.append(df)
all_df = pd.concat(df_list, axis=1)

# log_score_list = []
# for i in range(1, len(file_list)+1):
#     score = all_df[f"score_{i}"].values
#     log_score = np.log(score + 1e-9)
#     log_score_list.append(log_score)
# log_score_list = np.array(log_score_list)
# merged_score = np.exp(np.mean(log_score_list, axis=0))
# merged_score = np.clip(merged_score, 0.0, 1.0)

score_list = []
for i in range(1, len(file_list)+1):
    score = all_df[f"score_{i}"].values
    score_list.append(score)
score_list = np.array(score_list)
merged_score = np.average(score_list, axis=0)
merged_score = np.clip(merged_score, 0.0, 1.0)


all_df["merged_score"] = merged_score
all_df = all_df.reset_index()
all_df[["filename", "merged_score"]].to_csv(
    "submission_merged.csv", index=False, header=False)
# %%%
# all_df
# # %%
# x = all_df[[f"score_{i}" for i in range(1, 5)]].max(
#     axis=1) - all_df[[f"score_{i}" for i in range(1, 5)]].min(axis=1)
# # %%
# temp = all_df[x > 0.5]
# temp2 = temp[all_df[x > 0.5]["merged_score"] < 0.5]
# %%
# count = 0
# for im in temp2["filename"].tolist():
#     count += 1
#     if count < 200:
#         continue
#     print(im)
#     im = cv2.imread(f"/media/data/gen_orig_clas/evaluation/{im}")
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     plt.imshow(im)
#     plt.show()

#     if count > 230:
#         break

# # %%
# all_df.loc[x > 0.5, "merged_score"] = all_df[x > 0.5]["score_5"]
# # %%
# all_df[x > 0.5]
# # %%
# all_df[["filename", "merged_score"]].to_csv(
#     "submission_merged.csv", index=False, header=False)
# # %%
