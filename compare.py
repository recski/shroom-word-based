import json
import sys

import numpy as np


def evaluate(pred_labels, ref_labels):
    corr = 0
    htp = 0
    hfp = 0
    hfn = 0
    for i, pred_label in enumerate(pred_labels):
        ref_label = ref_labels[i]
        if pred_label == ref_label:
            corr += 1
            if pred_label == 'Hallucination':
                htp += 1
        elif pred_label == 'Hallucination':
            hfp += 1
        elif ref_label == 'Hallucination':
            hfn += 1
        else:
            assert False

    acc = corr / len(pred_labels)
    print(f"{acc=}")
    print(f"{htp=}")
    print(f"{hfp=}")
    print(f"{hfn=}")
    prec = htp / (htp + hfp)
    rec = htp / (htp + hfn)
    print(f"{prec=}")
    print(f"{rec=}")



def get_acc_t(pred_scores, ref_labels, t):
    corr = 0
    for i, score in enumerate(pred_scores):
        pred_label = "Hallucination" if score >= t else "Not Hallucination"
        ref_label = ref_labels[i]
        if pred_label == ref_label:
            corr += 1
    return corr / len(pred_scores)


def print_outliers(pred_data, ref_data, pred_scores, ref_scores):
    errs = [(abs(score - ref_scores[i]), i) for i, score in enumerate(pred_scores)]
    for err, i in sorted(errs)[-20:]:
        print(f'{err}\t{pred_scores[i]}\t{ref_scores[i]}')
        print(ref_data[i])
        print()

def main():
    pred_fn, ref_fn = sys.argv[1:3]
    with open(pred_fn) as f:
        pred_data = json.loads(f.read())
    with open(ref_fn) as f:
        ref_data = json.loads(f.read())

    pred_scores, ref_scores, pred_labels, ref_labels = [], [], [], []
    for i, line in enumerate(pred_data):
        pred_scores.append(line["p(Hallucination)"])
        pred_labels.append(line["label"])
        ref_labels.append(ref_data[i]["label"])
        ref_scores.append(ref_data[i]["p(Hallucination)"])

    for t in np.arange(0.0, 1.0, 0.025):
        acc = get_acc_t(pred_scores, ref_labels, t)
        print(f"t: {t}, acc: {acc:.2f}")

    evaluate(pred_labels, ref_labels)

    print_outliers(pred_data, ref_data, pred_scores, ref_scores)

if __name__ == "__main__":
    main()
