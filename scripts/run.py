import os 
from src.utils.core import claim_memory, set_seed
from datasets import disable_caching
from argparse import ArgumentParser
from src.repair.few_shot.FewShotRepair import FewShotRepair
from src.repair.sft.FinetunedRepair import FinetunedRepair
from src.repair.rl.RL import RL

from src.utils.files import json2data, read_config

def parse_args():
    parser = ArgumentParser(description="Running experiments")
    parser.add_argument("--config", required=True,
                        help="Path towards the configuration file")
    parser.add_argument("--train", action='store_true',
                        help="Train a model")
    parser.add_argument("--train_config", default="",
                        help="Train a model on specified dataset")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the model")
    parser.add_argument("--evaluate_config", default="",
                        help="Test a model on a specified configuration of a dataset.")
    parser.add_argument('--test_run',
                        help="Whether to do a test run to ensure the pipeline works without issues",
                        action="store_true")
    parser.add_argument('--experiment', choices=["FewShotRepair", "FinetunedRepair", "RL"],
                        help="Which experiment to run.")

    return parser.parse_args()

def main():
    args = parse_args()
    config = read_config(args.config)
    set_seed(config.seed)

    # for preprocessing without launching 
    experiment = globals()[args.experiment](config, args.test_run)

    if args.train: train(config, args)
    if args.evaluate: evaluate(config, args)
        
    
def train(config, args):
    if os.path.isfile(args.train_config):
        print("Updating configuration for training")
        config.dataset = read_config(args.train_config)
        print("Configuration being run", config.dataset)

    claim_memory()
    experiment = globals()[args.experiment](config, args.test_run)
    experiment.train()


def evaluate(config, args):
    if os.path.isfile(args.evaluate_config):
        print("Updating configuration for evaluation")
        config.dataset = read_config(args.evaluate_config)
        print("Configuration being run", config.dataset)

    claim_memory()
    experiment = globals()[args.experiment](config, args.test_run)
    experiment.evaluate()
    

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = '1'
    main()
