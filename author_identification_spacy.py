# -*- coding: utf-8 -*-
"""
Spooky Author Identification on the kaggle dataset
https://www.kaggle.com/c/spooky-author-identification
"""

import copy
import pandas as pd
import spacy
from spacy.util import minibatch, compounding

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def main():
    train_data = pre_process(train)
    nlp = model(train_data)
    evaluation(nlp)
    output(nlp)

def pre_process(train):
    train = pd.DataFrame(data=(train['text'], train['author']))
    train = train.transpose()
    train_texts = train['text']
    train_cats = train['author']
    test_texts = test['text']
    cats_values = train_cats.unique()
    labels_default = dict((v, 0) for v in cats_values)
    train_data = []
    for i, column in train.iterrows():
        label_values = copy.deepcopy(labels_default)
        label_values[column['author']] = 1
        train_data.append((str(column['text']), {"cats": label_values}))
    return train_data

def model(train_data):
    nlp = spacy.blank("en")
    textcat = nlp.create_pipe(
        "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
    )
    nlp.add_pipe(textcat, last=True)
    textcat.add_label("EAP")
    textcat.add_label("HPL")
    textcat.add_label("MWS")
    assert "EAP" in textcat.labels
    assert "HPL" in textcat.labels
    assert "MWS" in textcat.labels
    cats = {"EAP": 'EAP', "HPL": 'HPL', "MWS": 'MWS'}
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        print("Training the model...")
        print("{:^5}\t".format("LOSS"))
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(20):
            losses = {}
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, cats = zip(*batch)
                nlp.update(texts, cats, sgd=optimizer, drop=0.2, losses=losses)
            print("{0:.3f}".format(losses["textcat"]))
    output_dir = "spooky_author_id_output"
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
    return nlp

def evaluation(nlp):
    test_text = "In his left hand was a gold snuff box, from which, as he capered down the hill, cutting all manner of fantastic steps, he took snuff incessantly with an air of the greatest possible self satisfaction."
    doc = nlp(test_text)
    print(test_text, doc.cats)

def output(nlp):
    test = pd.read_csv('test.csv')
    test_data = []
    for i, column in test.iterrows():
        test_data.append((str(column['text']), column['id']))
    authors = sorted(nlp.get_pipe('textcat').labels)
    output = [['id'] + authors]
    for doc, id_ in nlp.pipe(test_data, as_tuples=True):
        scores = [str(doc.cats.get(author, 0.0)) for author in authors]
        output.append([id_] + scores)
    print(output[0], output[1])
    with open('output.csv', 'w') as file:
        lines = '\n'.join(','.join(row) for row in output)
        file.write(lines)
    print("Finished")

if __name__ == '__main__':
    main()
