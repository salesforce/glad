import json
import logging
import os
from pprint import pformat
from importlib import import_module
from vocab import Vocab
from dataset import Dataset, Ontology
from preprocess_data import dann


def load_dataset(splits=('train', 'dev', 'test')):
    with open(os.path.join(dann, 'ontology.json')) as f:
        ontology = Ontology.from_dict(json.load(f))
    with open(os.path.join(dann, 'vocab.json')) as f:
        vocab = Vocab.from_dict(json.load(f))
    with open(os.path.join(dann, 'emb.json')) as f:
        E = json.load(f)
    dataset = {}
    for split in splits:
        with open(os.path.join(dann, '{}.json'.format(split))) as f:
            logging.warn('loading split {}'.format(split))
            dataset[split] = Dataset.from_dict(json.load(f))

    logging.info('dataset sizes: {}'.format(pformat({k: len(v) for k, v in dataset.items()})))
    return dataset, ontology, vocab, E


def get_models():
    return [m.replace('.py', '') for m in os.listdir('models') if not m.startswith('_') and m != 'model']


def load_model(model, *args, **kwargs):
    Model = import_module('models.{}'.format(model)).Model
    model = Model(*args, **kwargs)
    logging.info('loaded model {}'.format(Model))
    return model
