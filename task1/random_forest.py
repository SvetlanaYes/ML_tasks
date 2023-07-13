"""
Implemented by : Svetlana Yesayan
Description:

    Script trains a model to classify web elements as product items using Random Forest and analyzes the results

Dependencies & Supported versions:

    See in requirements.txt

Usage:

    See details in README.md

"""
try:
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from collections import OrderedDict
    import warnings
    import sys
    import argparse
except ImportError as exception:
    print("You should import certain libraries")
    exit(0)


warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.")

RANDOM_STATE=42


def argument_parser():
    parser = argparse.ArgumentParser("Script trains a model to classify web elements as product items using Random Forest and analyzes the results")
    parser.add_argument("-d", "--dataset", type=str, default="data/task1.csv", required=False, help='Path to dataset in .csv format.')
    return parser.parse_args()


def preprocess_data(filename):
    df = pd.read_csv(filename)
    df = df.replace('[]', 0)
    df.dropna(how="all", inplace=True)
    df.dropna(axis="columns", how="all", inplace=True)
    df = df.fillna(0)
    df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).apply(lambda x: x.astype('category').cat.codes)
    y = df["is_shop"]
    df.drop("is_shop", axis=1, inplace=True)
    return df, y


def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


def plot_results(error_rate, min_estimators, max_estimators):
    for i, (label, clf_err) in enumerate(error_rate.items()):
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()


def get_best_model(error_rate, min_estimators, max_estimators):
    errors = [0, 0, 0]
    for i, (_, clf_err) in enumerate(error_rate.items()):
        _, ys = zip(*clf_err)
        errors[i] = sum(ys) / len(ys)
    plot_results(error_rate, min_estimators, max_estimators)
    return errors.index(min(errors))


def best_model_by_oob_error(ensemble_clfs, X_train, y_train):
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 15
    max_estimators = 150

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1, 5):
            clf.set_params(n_estimators=i)
            clf.fit(X_train, y_train)

            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    return ensemble_clfs[get_best_model(error_rate, min_estimators, max_estimators)][1]



def main(args):
    X, y = preprocess_data(args.dataset)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    ensemble_clfs = [
        (
            "RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(
                warm_start=True,
                max_features="log2",
                oob_score=True,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "RandomForestClassifier, max_features=None",
            RandomForestClassifier(
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    model = best_model_by_oob_error(ensemble_clfs, X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification report: \n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    args = argument_parser()
    main(args)
