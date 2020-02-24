from argparse import ArgumentParser
from inference import infer


parser = ArgumentParser()

parser.add_argument("modelname", help="name of model to use")
parser.add_argument("imagepath", help="relative path to image")
args = parser.parse_args()

try:
    infer(args.image, args.model)
except:
    print("INVALID ARGS::- path is wrong or model is missing")