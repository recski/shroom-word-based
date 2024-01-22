from nltk.corpus import stopwords

from similarity import levenshtein, wn_sim_wup


METHODS = {"levenshtein": levenshtein, "wordnet": wn_sim_wup}

THRESHOLDS = {"levenshtein": 0.275, "wordnet": 0.6}


def sim_norm(x):
    return 1 / (1 + x)


NORMALIZATION = {"levenshtein": sim_norm, "wordnet": lambda x: x, "paragram": lambda x: x}


def word_alignment(line, metric):

    method = METHODS[metric]
    threshold = THRESHOLDS[metric]
    norm_fnc = NORMALIZATION[metric]

    sw = set(stopwords.words("english"))

    def get_toks(text):
        toks = [word["lemma"] for sen in text["nlp"] for word in sen]
        return [tok for tok in toks if tok not in sw]

    hyp_toks = get_toks(line["hyp"])
    tgt_toks = get_toks(line["tgt"])
    src_toks = get_toks(line["src"])

    if line["ref"] == "tgt":
        ref_toks = tgt_toks
    elif line["ref"] == "src":
        ref_toks = src_toks
    elif line["ref"] == "either":
        if line["task"] == "MT":
            ref_toks = tgt_toks
        else:
            ref_toks = src_toks + tgt_toks

    sims = []
    for tok in hyp_toks:
        sims.append(max(norm_fnc(method(tok, tok2)) for tok2 in ref_toks))
    sims = [sim for sim in sims if sim is not None]
    if len(sims) == 0:
        raise ValueError(f"no similarities for sentence pair: {hyp_toks}, {ref_toks}")
    avg_sim = sum(sims) / len(sims)
    score = 1 - avg_sim
    label = "Hallucination" if score >= threshold else "Not Hallucination"

    return label, score
