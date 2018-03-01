from keras.models import load_model
from keras.utils import plot_model
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("model", help="model file path")
parser.add_argument("out_dir")
args = parser.parse_args()
print(args)

model = load_model(args.model)
plot_model(model, show_shapes=True, to_file=os.path.join(args.out_dir,'model.png'))


