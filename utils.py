def preprocess_data(data, nlp):
    new_data = []
    for line in data:
        new_line = {}
        for key, value in line.items():
            if key in ('hyp', 'src', 'tgt'):
                assert isinstance(value, str), value
                doc = nlp(value)
                new_line[key] = {'text': value, 'nlp': doc.to_dict()}
            else:
                new_line[key] = value
        new_data.append(new_line)

    return new_data
