from similarity import levenshtein

METHODS = {"levenshtein": levenshtein}

THRESHOLDS = {"levenshtein": 0.5}

def word_alignment(line, metric):
    
    method = METHODS[metric]
    threshold = THRESHOLDS[metric]

    hyp_toks = [word['text'] for sen in line['hyp']['nlp'] for word in sen]
    tgt_toks = [word['text'] for sen in line['tgt']['nlp'] for word in sen]
    src_toks = [word['text'] for sen in line['src']['nlp'] for word in sen]

    if line['ref'] == 'tgt':
        ref_toks = tgt_toks
    elif line['ref'] == 'src':
        ref_toks = src_toks
    elif line['ref'] == 'either':
        if line['task'] == 'MT':
            ref_toks = tgt_toks
        else:
            ref_toks = src_toks + tgt_toks

    sims = []
    for tok in hyp_toks:
        sims.append(max(method(tok, tok2) for tok2 in ref_toks))

    avg_sim = sum(sims) / len(sims)

    label = 'Hallucination' if avg_sim >= threshold else 'Not Hallucination'

    return label, avg_sim
    
