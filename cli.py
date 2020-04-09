from argparse import ArgumentParser
from inference import infer


parser = ArgumentParser()

parser.add_argument("modelname", help="name of model to use")
parser.add_argument("imagepath", help="relative path to image")
args = parser.parse_args()

try:
    infer(args.imagepath, args.modelname)
except:
    print("Something BAD happened!!!")