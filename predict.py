import argparse
import json
import logging
import sys

from nltk.corpus import stopwords
from tqdm import tqdm

SW = set(stopwords.words("english"))


def sim_norm(x):
    return 1 / (1 + x)


NORMALIZATION = {"levenshtein": sim_norm, "wordnet": lambda x: x, "paragram": lambda x: x}


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--method", type=str, required=True)
    parser.add_argument("-t", "--threshold", type=float, required=True)
    parser.add_argument("-g", "--global-scoring", type=str, required=True)
    parser.add_argument("-s", "--stopword-removal", default=False, action='store_true')
    return parser.parse_args()


def avg(sims):
    return sum(sims) / len(sims)


def harmonic(sims):
    if any(s == 0 for s in sims):
        return 0
    return len(sims) / sum(1 / sim for sim in sims)


def transform_sims(sims, hyp_toks, ref_toks, norm_fnc, stopword_removal):
    t_sims = []
    for i in range(len(hyp_toks)):
        t_sims.append([])
        if stopword_removal and hyp_toks[i] in SW:
            continue
        for j in range(len(ref_toks)):
            if stopword_removal and ref_toks[j] in SW:
                continue
            t_sims[-1].append(norm_fnc(sims[i][j]))

    return t_sims


def align_min_best(sims):
    return min([max(hyp_tok_sims) for hyp_tok_sims in sims if len(hyp_tok_sims) > 0])


def align_harmonic_best(sims):
    return harmonic([max(hyp_tok_sims) for hyp_tok_sims in sims if len(hyp_tok_sims) > 0])


def align_avg_best(sims):
    return avg([max(hyp_tok_sims) for hyp_tok_sims in sims if len(hyp_tok_sims) > 0])


GLOBAL_SCORING = {
    "avg_best": align_avg_best,
    "harmonic_best": align_harmonic_best,
    "min_best": align_min_best
}


def main():
    args = get_args()
    logging.basicConfig(
        format="%(asctime)s : "
        + "%(module)s (%(lineno)s) - %(levelname)s - %(message)s"
    )
    logging.getLogger().setLevel(logging.DEBUG)
    global_fnc = GLOBAL_SCORING[args.global_scoring]
    norm_fnc = NORMALIZATION[args.method]
    assert 0 <= args.threshold <= 1
    output = []
    for raw_line in tqdm(sys.stdin):
        line = json.loads(raw_line)
        sims = line['sims'][args.method]
        hyp_toks = line['hyp_toks']
        ref_toks = line['ref_toks']

        transformed_sims = transform_sims(sims, hyp_toks, ref_toks, norm_fnc, args.stopword_removal)
        global_sim = global_fnc(transformed_sims)

        score = 1 - global_sim
        label = "Hallucination" if score >= args.threshold else "Not Hallucination"
        output.append({"id": line.get("id"), "label": label, "p(Hallucination)": score})

    print(json.dumps(output))


if __name__ == "__main__":
    main()
