from nltk.corpus import wordnet as wn

from embedding import TSVEmbedding

PARAGRAM = TSVEmbedding('vectors/paragram_300_sl999/paragram_300_sl999.txt', tab_first=False)
# PARAGRAM = TSVEmbedding('vectors/paragram_300_sl999/paragram_sample.txt', tab_first=False)


def paragram(w1, w2):
    sim = PARAGRAM.get_sim(w1, w2)
    if sim is None:
        return 0
    return sim


def wn_sim_wup(w1, w2):
    sims = [s1.wup_similarity(s2) for s1 in wn.synsets(w1) for s2 in wn.synsets(w2)]
    if len(sims) == 0:
        return 0
    return max(sims)


def levenshtein(s1, s2):
    """
    from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = (
                previous_row[j + 1] + 1
            )  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
