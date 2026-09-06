import argparse
from omegaconf import OmegaConf

from imitation_src.train import train_model
from imitation_src.eval import eval_model
from imitation_src.visualize import visualize_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Argument for car racing train/eval/vis')
    parser.add_argument("--mode", type=str, default = "train", choices = ['train', 'eval', 'vis'], help = 'Training, evaluation or visualization')
    parser.add_argument("--config", type=str, default = "./config/simple.yaml", help="Path to config file")
    args = parser.parse_args()
    
    configs = OmegaConf.load(args.config) # https://omegaconf.readthedocs.io/en/latest/usage.html#from-a-yaml-file

    if args.mode == 'train':
        train_model(configs)
    elif args.mode == 'eval':
        eval_model(configs)
    else:
        visualize_model(configs)
        