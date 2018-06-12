import os
import json
import logging
from argparse import ArgumentParser, Namespace
from pprint import pprint
from utils import load_dataset, load_model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dsave', help='save location of model')
    parser.add_argument('--split', help='split to evaluate on', default='dev')
    parser.add_argument('--gpu', type=int, help='gpu to use', default=None)
    parser.add_argument('--fout', help='optional save file to store the predictions')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(os.path.join(args.dsave, 'config.json')) as f:
        args_save = Namespace(**json.load(f))
        args_save.gpu = args.gpu
    pprint(args_save)

    dataset, ontology, vocab, Eword = load_dataset()

    model = load_model(args_save.model, args_save, ontology, vocab)
    model.load_best_save(directory=args.dsave)
    if args.gpu is not None:
        model.cuda(args.gpu)

    logging.info('Making predictions for {} dialogues and {} turns'.format(len(dataset[args.split]), len(list(dataset[args.split].iter_turns()))))
    preds = model.run_pred(dataset[args.split], args_save)
    pprint(dataset[args.split].evaluate_preds(preds))

    if args.fout:
        with open(args.fout, 'wt') as f:
            # predictions is a list of sets, need to convert to list of lists to make it JSON serializable
            json.dump([list(p) for p in preds], f, indent=2)
