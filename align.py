import json
import logging
import sys

import stanza
from tuw_nlp.text.pipeline import CachedStanzaPipeline

from utils import preprocess_data
from similarity import levenshtein, paragram, wn_sim_wup

METHODS = {"levenshtein": levenshtein, "wordnet": wn_sim_wup, "paragram": paragram}

THRESHOLDS = {"levenshtein": 0.275, "wordnet": 0.6, "paragram": 0.4}


def sim_norm(x):
    return 1 / (1 + x)


NORMALIZATION = {"levenshtein": sim_norm, "wordnet": lambda x: x, "paragram": lambda x: x}


def avg(sims):
    return sum(sims) / len(sims)


GLOBAL_SIM = {"levenshtein": min, "wordnet": min, "paragram": min}

# GLOBAL_SIM = {"levenshtein": avg, "wordnet": avg, "paragram": avg}


def get_toks_from_line(line):

    def get_toks(text):
        return [word["lemma"] for sen in text["nlp"] for word in sen]

    hyp_toks = get_toks(line["hyp"])
    tgt_toks = get_toks(line["tgt"])
    src_toks = get_toks(line["src"])

    if 'ref' not in line:
        if line["task"] in ("MT", "DM"):
            ref_toks = tgt_toks
        else:
            assert line['task'] == 'PG'
            ref_toks = src_toks + tgt_toks

    elif line["ref"] == "tgt":
        ref_toks = tgt_toks
    elif line["ref"] == "src":
        ref_toks = src_toks
    elif line["ref"] == "either":
        if line["task"] == "MT":
            ref_toks = tgt_toks
        else:
            ref_toks = src_toks + tgt_toks

    return hyp_toks, ref_toks


def word_alignment(line, metric):

    method = METHODS[metric]
    threshold = THRESHOLDS[metric]
    norm_fnc = NORMALIZATION[metric]
    global_sim_fnc = GLOBAL_SIM[metric]

    hyp_toks, ref_toks = get_toks_from_line(line)
    logging.debug(f'{hyp_toks=}, {ref_toks=}')

    sims = []
    for tok in hyp_toks:
        best = max((norm_fnc(method(tok, tok2)), tok2) for tok2 in ref_toks)
        sim = best[0]
        logging.debug(f'{tok=}, {best=}')
        if sim is not None:
            sims.append(sim)

    if len(sims) == 0:
        raise ValueError(f"no similarities for sentence pair: {hyp_toks}, {ref_toks}")

    global_sim = global_sim_fnc(sims)
    score = 1 - global_sim
    label = "Hallucination" if score >= threshold else "Not Hallucination"

    return label, score


def test_alignment():
    logging.basicConfig(
        format="%(asctime)s : "
        + "%(module)s (%(lineno)s) - %(levelname)s - %(message)s"
    )
    logging.getLogger().setLevel(logging.DEBUG)
    metric = sys.argv[1]
    print(f'{metric=}')
    data = json.load(sys.stdin)
    nlp_pipeline = stanza.Pipeline("en", processors="tokenize,pos,lemma")
    nlp_cache = "nlp_cache.json"
    with CachedStanzaPipeline(nlp_pipeline, nlp_cache) as nlp:
        preprocessed_data = preprocess_data(data, nlp)
    for line in preprocessed_data:
        label, score = word_alignment(line, metric)
        print(f'{label=}')
        print(f'{score=}')


if __name__ == "__main__":
    test_alignment()
