import json
import logging
import sys

import stanza
from tqdm import tqdm
from tuw_nlp.text.pipeline import CachedStanzaPipeline

from align import METHODS, get_toks_from_line
from utils import preprocess_data


def main():
    logging.basicConfig(
        format="%(asctime)s : "
        + "%(module)s (%(lineno)s) - %(levelname)s - %(message)s"
    )
    logging.getLogger().setLevel(logging.DEBUG)
    data = json.load(sys.stdin)
    nlp_pipeline = stanza.Pipeline("en", processors="tokenize,pos,lemma")
    nlp_cache = "nlp_cache.json"
    with CachedStanzaPipeline(nlp_pipeline, nlp_cache) as nlp:
        for line in tqdm(preprocess_data(data, nlp)):
            hyp_toks, ref_toks = get_toks_from_line(line)
            line['hyp_toks'] = hyp_toks
            line['ref_toks'] = ref_toks
            line['sims'] = {
                method: [
                    [fnc(hyp_tok, ref_tok) for ref_tok in ref_toks]
                    for hyp_tok in hyp_toks
                ]
                for method, fnc in METHODS.items()
            }
            print(json.dumps(line))


if __name__ == "__main__":
    main()
