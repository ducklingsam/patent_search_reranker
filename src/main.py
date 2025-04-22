import argparse
import os
from loguru import logger
from .embedder import train_embedder
from .reranker import PatentReranker
from .evaluate import evaluate_manual
from .experiments import compare_methods, ablation_study, hyperparameter_tuning

from utils import download_model


def main():
    model_path = 'models/contrastive-rubert/model.safetensors'

    if not os.path.isfile(model_path):
        model_url = 'https://your-link.com/model.safetensors'
        logger.info('No model.safetensors found in models/contrastive-rubert. Downloading it now...')
        download_model(model_url, model_path)

    parser = argparse.ArgumentParser(description="Pipeline for Patent Search Re-ranker")
    sub = parser.add_subparsers(dest='cmd')

    e1 = sub.add_parser('embed')
    e1.add_argument('--gold', required=True)
    e1.add_argument('--output', required=True)
    e1.add_argument('--epochs', type=int, default=1)

    tr = sub.add_parser('train')
    tr.add_argument('--gold', required=True)
    tr.add_argument('--output', required=True)

    ev = sub.add_parser('eval')
    ev.add_argument('--manual', required=True)
    ev.add_argument('--model', required=True)

    exp = sub.add_parser('experiment')
    exp.add_argument('--gold', required=True)
    exp.add_argument('--manual', required=True)
    exp.add_argument('--model', required=True)

    args = parser.parse_args()

    if args.cmd == 'embed':
        train_embedder(args.gold, output_path=args.output, epochs=args.epochs)
    elif args.cmd == 'train':
        PatentReranker().train(args.gold, args.output)
    elif args.cmd == 'eval':
        evaluate_manual(args.manual, args.model)
    elif args.cmd == 'experiment':
        print("Running method comparison...")
        compare_methods(args.manual, args.model)
        print("\nRunning ablation study...")
        ablation_study(args.gold, 'ablation_results.csv')
        print("\nRunning hyperparameter tuning...")
        hyperparameter_tuning(args.gold, 'hyperparam_results.csv')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()