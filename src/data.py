import os


def get_data_dirs(train_dir="data/train", test_dir="data/test"):
    # dataset ships in the repo
    if not os.path.isdir(train_dir):
        raise SystemExit(f"missing {train_dir}/")
    return train_dir, test_dir
