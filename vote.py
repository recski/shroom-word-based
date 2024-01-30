import json
import sys

# THRESHOLD = 0.51
THRESHOLD = 0.49


def label_to_feat(label):
    if label == "Hallucination":
        return 1
    elif label == "Not Hallucination":
        return 0
    else:
        assert False


def main():
    data = {}
    names = []
    for fn in sys.argv[1:]:
        name = fn.split("/")[-2]
        if name == "gold":
            continue
        names.append(name)
        print(f"loading {name}")
        with open(fn) as f:
            d = json.load(f)
            data[name] = d

    lengths = set(len(data[name]) for name in data)
    assert len(lengths) == 1, f"files have different lengths, {lengths=}"
    length = lengths.pop()

    output = []

    for i in range(length):
        line_id = data["levenshtein"][i]["id"]
        count = 0
        for name in names:
            label = data[name][i]["label"]
            count += label_to_feat(label)
            if line_id is not None:
                assert (
                    data[name][i]["id"] == line_id
                ), f'{i=}, {line_id=}, {name=}, {data[name][i]["id"]=}'

        ratio = count / len(names)
        label = "Hallucination" if ratio >= THRESHOLD else "Not Hallucination"
        output.append({"id": line_id, "label": label, "p(Hallucination)": ratio})

    with open("output.json", "w") as f:
        f.write(json.dumps(output))


if __name__ == "__main__":
    main()
