import pandas as pd

dfs = []
for i in range(1, 5):
    anno_file = f"/media/data/gen_orig_clas/train_{i}.csv"
    df = pd.read_csv(anno_file, header=None)
    df.columns = ["path", "category"]
    df.iloc[:, 0] = f"train_{i}/" + df.iloc[:, 0]
    df["group"] = i
    dfs.append(df)
dfs = pd.concat(dfs)

dfs.to_csv("/media/data/gen_orig_clas/all.csv", index=None)
