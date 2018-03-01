import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("out")
args = parser.parse_args()

import coremltools
coreml_model = coremltools.converters.keras.convert(args.model,
        input_names = 'image',
        image_input_names = 'image',
        image_scale = 0.00392156863)

import os
coreml_model.save(os.path.join(args.out, 'SRCNN.mlmodel'))

