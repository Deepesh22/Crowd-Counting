from argparse import ArgumentParser
from inference import Infer


parser = ArgumentParser()

parser.add_argument("modelname", help="name of model to use")
parser.add_argument("imagepath", help="relative path to image")
parser.add_argument("--use_gpu", help="use gpu or not", nargs="?", default=False, const=True, type = bool)
args = parser.parse_args()

infer = Infer(args.use_gpu)

try:
    infer.infer(args.imagepath, args.modelname)
except:
    print("Something BAD happened!!!")