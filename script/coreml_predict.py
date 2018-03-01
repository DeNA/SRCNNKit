import coremltools
import numpy as np
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("model", help="model file path")
parser.add_argument("input", help="input patch image")
parser.add_argument("output", help="output patch image")
args = parser.parse_args()

model = coremltools.models.MLModel(args.model)
img = Image.open(args.input).convert('RGB')
x = {'image': img}
res = model.predict(x)

out = np.asarray(res['output1'] * 255., np.uint8)
print(out.shape)
out = np.rollaxis(out, 0, 3)
print(out.shape)
outimg = Image.fromarray(out)
outimg.save(args.output)

