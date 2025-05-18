import argparse
import os
from loguru import logger
from .embedder import train_embedder
from .reranker import PatentReranker
from .evaluate import evaluate_manual
from .experiments import compare_methods, ablation_study, hyperparameter_tuning
from .explainability import analyze_explainability
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import download_model


def main():
    model_path = 'models/contrastive-rubert/model.safetensors'

    if not os.path.isfile(model_path):
        model_url = 'https://limewire.com/d/gbCI2#mQvObSb6Mz'
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

    out = sub.add_parser('explain')
    out.add_argument('--manual', required=True)
    out.add_argument('--model', required=True)
    out.add_argument('--out', required=True)

    args = parser.parse_args()

    if args.cmd == 'embed':
        train_embedder(args.gold, output_path=args.output, epochs=args.epochs)
    elif args.cmd == 'train':
        PatentReranker().train(args.gold, args.output)
    elif args.cmd == 'eval':
        evaluate_manual(args.manual, args.model)
    elif args.cmd == 'experiment':
        print("Running all experiments in parallel...\n")

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(compare_methods, args.manual, args.model): 'Method Comparison',
                executor.submit(ablation_study, args.gold, 'ablation_results_new.csv'): 'Ablation Study',
                # executor.submit(hyperparameter_tuning, args.gold, 'hyperparam_results.csv'): 'Hyperparameter Tuning'
            }

            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    future.result()
                    print(f"\n✅ {task_name} completed.")
                except Exception as e:
                    print(f"\n❌ {task_name} failed with error: {e}")

    elif args.cmd == 'explain':
        analyze_explainability(args.manual, args.model, args.out)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()