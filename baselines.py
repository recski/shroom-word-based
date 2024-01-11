import json
import os
import random
import sys

import stanza
from tuw_nlp.text.pipeline import CachedStanzaPipeline

from utils import preprocess_data
from align import word_alignment


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def edit_distance(line):
    return word_alignment(line, "levenshtein")


def always_yes(line):
    p = random.choice(("0.6", "0.8", "1.0"))
    return "Hallucination", p


def always_no(line):
    p = random.choice(("0.0", "0.2", "0.4"))
    return "Not Hallucination", p


SYSTEMS = {
    "always_no": always_no,
    "always_yes": always_yes,
    "edit_distance": edit_distance,
}


def main():
    aware_fn, agnostic_fn = sys.argv[1:]
    fns = {"aware": aware_fn, "agnostic": agnostic_fn}

    nlp_pipeline = stanza.Pipeline("en", processors="tokenize,pos")
    nlp_cache = "nlp_cache.json"
    with CachedStanzaPipeline(nlp_pipeline, nlp_cache) as nlp:
        for data_type in ("aware", "agnostic"):
            with open(fns[data_type]) as f:
                data = json.load(f)

            preprocessed_data = preprocess_data(data, nlp)

            for name, function in SYSTEMS.items():
                output = []
                for line in preprocessed_data:
                    label, p = function(line)
                    output.append(
                        {"id": line.get("id"), "label": label, "p(Hallucination)": p}
                    )
                out_dir = f"output/{name}/dev"
                ensure_dir(out_dir)
                with open(
                    os.path.join(out_dir, f"val.model-{data_type}.json"), "w"
                ) as f:
                    f.write(json.dumps(output))


if __name__ == "__main__":
    main()
