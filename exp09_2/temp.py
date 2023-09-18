#%%
import pandas as pd


pred = pd.read_csv("../work_dirs/exp09/val_exp09_val_group_1.csv", header=None)
gt = pd.read_csv("/media/data/gen_orig_clas/train_1.csv",header=None)
pred.columns = ["filename", "pred"]
gt.columns = ["filename", "true"]

# %%
df = gt.merge(pred, on="filename")
# %%
df["pred_binary"] = (df["pred"] >= 0.5).astype(int)
# %%
df[df["pred_binary"]!=df["true"]]
# %%
