from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import numpy
import math
import os
from keras.models import load_model
import cv2
from PIL import Image

input_size = 200
label_size = 200

def setup_session():
    import tensorflow as tf
    from keras.backend import tensorflow_backend
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)

def predict(srcnn_model, input_path, out_dir, use_coreml):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    dst = numpy.copy(img)
    h,w,c = img.shape
    blk = numpy.zeros((h+12,w+12,c))
    blk[6:6+h, 6:6+w,:] = img
    dst_base = numpy.copy(blk)
    for y in range(0, h+12, label_size):
        for x in range(0, w+12, label_size):
            part = blk[y:y+input_size,x:x+input_size]
            in_path = "out/pred_in/%d-%d.png" % (y,x)
            ba_path = "out/pred_base/%d-%d.png" % (y,x)
            ot_path = "out/pred_out/%d-%d.png" % (y,x)
            #cv2.imwrite(in_path, part)

            base = make_base(part, ba_path, ot_path)
            dst_base[y:y+input_size,x:x+input_size] = base

            out = exec_pred(srcnn_model, part, ba_path, ot_path, use_coreml)
            dst[y:y+label_size,x:x+label_size] = out
    base_path = os.path.join(out_dir, 'base.png')
    out_path = os.path.join(out_dir, 'out.png')
    print(base_path, out_path)
    cv2.imwrite(base_path, dst_base[6:-6,6:-6])
    cv2.imwrite(out_path, dst)

def make_base(img, ba_path, ot_path):
    h,w,c = img.shape
    if h < input_size or w < input_size:
        return
    img = img.astype(numpy.uint8)
    shape = img.shape
    img = cv2.resize(img, (int(shape[1] / 2), int(shape[0] / 2)), cv2.INTER_CUBIC)
    img = cv2.resize(img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    return img

def exec_pred(srcnn_model, img, ba_path, ot_path, use_coreml):
    h,w,c = img.shape
    if h < input_size or w < input_size:
        return
    img = img.astype(numpy.uint8)
    shape = img.shape
    img = cv2.resize(img, (int(shape[1] / 2), int(shape[0] / 2)), cv2.INTER_CUBIC)
    img = cv2.resize(img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    cv2.imwrite(ba_path, img)

    Y = numpy.zeros((1, img.shape[0], img.shape[1], c), dtype=float)
    Y[0] = img.astype(float) / 255.
    #print(Y.shape)
    pre = run_pred(srcnn_model, Y, use_coreml)
    #print('pred_shape', pre.shape)
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(numpy.uint8)
    #img[6: -6, 6: -6] = pre[0]
    #img = img[6:-6, 6:-6]
    #cv2.imwrite(ot_path, img)
    return pre 

def run_pred(model, Y, use_coreml):
    if use_coreml:
        _, h, w, c = Y.shape
        img = Y.reshape((h,w,c))
        img = Image.fromarray(img)
        x = {'image': img}
        res = model.predict(x)
        out = np.asarray(res['output1'] * 255., np.uint8)
        out = np.rollaxis(out, 0, 3)
        return out
    else:
        return model.predict(Y, batch_size=1) * 255.

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model file path")
    parser.add_argument("input", help="data dir")
    parser.add_argument("out_dir", help="output dir")
    parser.add_argument("-coreml", help="FIXME:use coreml model", action='store_true')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.coreml:
        import coremltools
        model = coremltools.models.MLModel(args.model)
        print(model)
    else:
        setup_session()
        model = load_model(args.model)

    predict(model, args.input, args.out_dir, args.coreml)
    print('fin')
