import os

COMPETITION = "ucsc-cse-144-spring-2026-final-project"


def get_data_dirs(train_dir="data/train", test_dir="data/test"):
    # grab from kaggle if it isn't already here
    if os.path.isdir(train_dir) and os.listdir(train_dir):
        return train_dir, test_dir
    import kagglehub
    try:
        path = kagglehub.competition_download(COMPETITION)
    except Exception as e:
        raise SystemExit(
            "kaggle download failed. put your kaggle.json at ~/.kaggle/kaggle.json, "
            f"or stage the data under {train_dir}/.\n{e}")
    return os.path.join(path, "train"), os.path.join(path, "test")
