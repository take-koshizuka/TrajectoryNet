import argparse
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams"]

parser = argparse.ArgumentParser("Continuous Normalizing Flow")
parser.add_argument('-config', '-c', help="the configuration file for training.", type=str, required=True)
parser.add_argument("-out_dir", '-o', help="the directory to output the analysis results.", type=str, required=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--no_display_loss", action="store_false")
