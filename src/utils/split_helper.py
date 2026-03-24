# imports
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit

from src.core.session import session

# import src.utils.preprocess_helper as pre


def split_data(data, config):
    train_size = config.get("train_size", 0.7)
    val_data = config.get("val_data", False)

    if val_data:
        test_size = (1 - float(train_size)) / 2
    else: 
        test_size = (1 - float(train_size))

    n = len(data)

    n_train = int(n * train_size)
    n_val = int(n * test_size)

    train_data = data[:n_train]

    if val_data:
        val_data = data[n_train:n_train+n_val]
        test_data = data[n_train+n_val:]

    else: 
        test_data = data[n_train:]
        val_data = []

    return train_data, test_data, val_data


def validation_split(model, X_train, y_train):
    random_state = session.parameters.get("random_state", 42)

    X_train2, X_val, y_train2, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )
    # train model on validation dataset
    model.fit(X_train2, y_train2)
    y_proba_val = model.predict_proba(X_val)[:, 1]

    return y_val, y_proba_val


def group_split(X, y, split_mode="group_kfold"):
    from src.core.mlflow_logger import get_experiment_logger
    
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    X, groups = make_feature_signature(X)

    random = session.parameters.get("random_state", 42)
    test_size = session.parameters.get("test_size", 0.2)

    if split_mode == "group_shuffle":
        n_splits = session.parameters.get("n_splits", None)

        if n_splits is None:
            raise ValueError(
                "[ERROR] No Value provided for n_splits. (split_mode: %s)", split_mode
            )

        event_logger.info(
            "Apply GroupShuffleSplit(n_splits=%s, test_size=%s)", n_splits, test_size
        )

        group_split = GroupShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random
        )

    elif split_mode == "group_kfold":
        n_folds = session.parameters.get("n_folds", None)
        # test_size = session.parameters.get("test_size", 0.2)

        if n_folds is None:
            raise ValueError(
                "[ERROR] No Value provided for n_folds. (split_mode: %s)", split_mode
            )

        event_logger.info(
            "Apply GroupKFold (n_folds=%s, test_size=%s)", n_folds, test_size
        )

        group_split = GroupKFold(n_splits=n_folds)

    for split_id, (train_idx, test_idx) in enumerate(group_split.split(X, y, groups)):
        # correct id to make it more readible
        split_id += 1

        event_logger.info("=== Split / Fold %s ===", split_id)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # pre-training check
        assert list(X_train.columns) == list(X_test.columns)
        pre.pretraining_checks(X_train, X_test, y_train, y_test)

        yield split_id, X_train, X_test, y_train, y_test
