#!/usr/bin/env python
import os
import json
import logging
import requests
from tqdm import tqdm
from vocab import Vocab
from embeddings import GloveEmbedding, KazumaCharEmbedding
from dataset import Dataset, Ontology


root_dir = os.path.dirname(__file__)
data_dir = os.path.join(root_dir, 'data', 'woz')


draw = os.path.join(data_dir, 'raw')
dann = os.path.join(data_dir, 'ann')

splits = ['dev', 'train', 'test']


def download(url, to_file):
    r = requests.get(url, stream=True)
    with open(to_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def missing_files(d, files):
    return not all([os.path.isfile(os.path.join(d, '{}.json'.format(s))) for s in files])


if __name__ == '__main__':
    if missing_files(draw, splits):
        if not os.path.isdir(draw):
            os.makedirs(draw)
        download('https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_train_en.json', os.path.join(draw, 'train.json'))
        download('https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_validate_en.json', os.path.join(draw, 'dev.json'))
        download('https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_test_en.json', os.path.join(draw, 'test.json'))

    if missing_files(dann, files=splits + ['ontology', 'vocab', 'emb']):
        if not os.path.isdir(dann):
            os.makedirs(dann)
        dataset = {}
        ontology = Ontology()
        vocab = Vocab()
        vocab.word2index(['<sos>', '<eos>'], train=True)
        for s in splits:
            fname = '{}.json'.format(s)
            logging.warn('Annotating {}'.format(s))
            dataset[s] = Dataset.annotate_raw(os.path.join(draw, fname))
            dataset[s].numericalize_(vocab)
            ontology = ontology + dataset[s].extract_ontology()
            with open(os.path.join(dann, fname), 'wt') as f:
                json.dump(dataset[s].to_dict(), f)
        ontology.numericalize_(vocab)
        with open(os.path.join(dann, 'ontology.json'), 'wt') as f:
            json.dump(ontology.to_dict(), f)
        with open(os.path.join(dann, 'vocab.json'), 'wt') as f:
            json.dump(vocab.to_dict(), f)

        logging.warn('Computing word embeddings')
        embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
        E = []
        for w in tqdm(vocab._index2word):
            e = []
            for emb in embeddings:
                e += emb.emb(w, default='zero')
            E.append(e)
        with open(os.path.join(dann, 'emb.json'), 'wt') as f:
            json.dump(E, f)
