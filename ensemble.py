import json
import sys


import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier


def label_to_feat(label):
    if label == "Hallucination":
        return 1
    elif label == "Not Hallucination":
        return 0
    else:
        assert False


def feat_to_label(feat):
    if feat == 1:
        return "Hallucination"
    elif feat == 0:
        return "Not Hallucination"
    else:
        assert False


def try_models(X, y):
    assert len(X) == len(y)

    models = {
        "nb": BernoulliNB(),
        "dt": DecisionTreeClassifier(random_state=0),
        "logreg": LogisticRegression(random_state=0),
    }

    for method, model in models.items():
        scores = cross_val_score(model, X, y, cv=10)
        avg_score = sum(scores) / len(scores)
        print(f"{method=}, {avg_score=}, {scores=}")
        model.fit(X, y)
        if method == "logreg":
            print(f"{model.coef_=}")
        if method == "nb":
            print(f"{model.feature_log_prob_=}")


def train_model(X, y):
    model = LogisticRegression(random_state=0)
    model.fit(X, y)
    dump(model, "model.joblib")


def use_model(X, ids):
    assert len(X) == len(ids)
    model = load("model.joblib")
    y = model.predict(X)
    print(f"{y.shape=}")
    p = model.predict_proba(X)
    print(f"{p.shape=}")
    output = []
    for i in range(len(ids)):
        output.append(
            {"id": ids[i], "label": feat_to_label(y[i]), "p(Hallucination)": p[i][1]}
        )

    with open("output.json", "w") as f:
        f.write(json.dumps(output))


def main():
    data = {}
    names = []
    for fn in sys.argv[1:]:
        name = fn.split("/")[-2]
        names.append(name)
        print(f"loading {name}")
        with open(fn) as f:
            d = json.load(f)
            data[name] = d

    lengths = set(len(data[name]) for name in data)
    assert len(lengths) == 1, f"files have different lengths, {lengths=}"
    length = lengths.pop()

    X, y, ids = [], [], []
    for i in range(length):
        X.append([])
        line_id = data["levenshtein"][i]["id"]
        ids.append(line_id)
        for name in names:
            label = data[name][i]["label"]
            if line_id is not None:
                assert (
                    data[name][i]["id"] == line_id
                ), f'{i=}, {line_id=}, {name=}, {data[name][i]["id"]=}'
            # p = data[name][i]['p(Hallucination)']
            feat = label_to_feat(label)
            if name == "gold":
                y.append(feat)
            else:
                X[-1].append(feat)

    X = np.array(X)
    print(f"{X.shape=}")
    y = np.array(y)
    print(f"{y.shape=}")
    print(f"{names=}")
    # try_models(X, y)
    # train_model(X, y)
    use_model(X, ids)


if __name__ == "__main__":
    main()
