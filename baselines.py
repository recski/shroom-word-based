import json
import os
import random
import sys


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def edit_dist(line):
    pass


def always_yes(line):
    p = random.choice(('0.6', '0.8', '1.0'))
    return "Hallucination", p


def always_no(line):
    p = random.choice(('0.0', '0.2', '0.4'))
    return "Not Hallucination", p


SYSTEMS = {"always_no": always_no, "always_yes": always_yes}


def main():
    aware_fn, agnostic_fn = sys.argv[1:]
    fns = {'aware': aware_fn, 'agnostic': agnostic_fn}
    for data_type in ('aware', 'agnostic'):
        with open(fns[data_type]) as f:
            data = json.load(f)

        for name, function in SYSTEMS.items():
            output = []
            for line in data:
                label, p = function(line)
                output.append({"id": line.get("id"), "label": label, "p(Hallucination)": p})
            out_dir = f"output/{name}/dev"
            ensure_dir(out_dir)
            with open(os.path.join(out_dir, f"val.model-{data_type}.json"), "w") as f:
                f.write(json.dumps(output))


if __name__ == "__main__":
    main()
