import os
# 実行中のファイルの絶対パスを取得
current_file_path = os.path.abspath(__file__)
# 絶対パスからディレクトリ名を取得
current_directory = os.path.basename(os.path.dirname(current_file_path))


class Config:
    exp = current_directory
    work_dirs = "../work_dirs"
    # train 設定
    val_group = 1
    num_epochs = 300
    train_batch_size = 8
    val_batch_size = 8
    learning_rate = 0.001 / train_batch_size
    seed = 0
    data_dir = "/media/data/gen_orig_clas"
    annotation_filename = "all.csv"
    eta_min = learning_rate/10
    # test 設定
    test_epoch = 300


if __name__ == "__main__":
    # クラス変数の一覧を取得
    class_variables = {var: getattr(Config, var) for var in dir(Config)
                       if not callable(getattr(Config, var))
                       and not var.startswith("__")}

    print(class_variables)
