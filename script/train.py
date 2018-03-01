from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from keras.optimizers import SGD, Adam
import numpy as np
import math
import pred
import os
import random
from os import listdir, makedirs
from os.path import isfile, join, exists
from PIL import Image

def model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='glorot_uniform', activation='relu', border_mode='same', bias=True, input_shape=(200, 200, 3)))
    SRCNN.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform', activation='relu', border_mode='same', bias=True))
    SRCNN.add(Conv2D(nb_filter=3, nb_row=5, nb_col=5, init='glorot_uniform', activation='linear', border_mode='same', bias=True))
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN

class MyDataGenerator(object):

    def flow_from_directory(self, input_dir, label_dir, batch_size=32):
        images = []
        labels = []
        while True:
            files = listdir(input_dir)
            random.shuffle(files)
            for f in files:
                images.append(self.load_image(input_dir, f))
                labels.append(self.load_image(label_dir, f))
                if len(images) == batch_size:
                    x_inputs = np.asarray(images)
                    x_labels = np.asarray(labels)
                    images = []
                    labels = []
                    yield x_inputs, x_labels

    def load_image(self, src_dir, f):
        X = np.asarray(Image.open(join(src_dir, f)).convert('RGB'), dtype='float32')
        X /= 255.
        return X

def train(log_dir, model_dir, train_dir, test_dir, eval_img):
    srcnn_model = model()
    print(srcnn_model.summary())

    datagen = MyDataGenerator()
    train_gen = datagen.flow_from_directory(os.path.join(
        train_dir, 'input'),
        os.path.join(train_dir, 'label'),
        batch_size = 10)

    val_gen = datagen.flow_from_directory(
        os.path.join(test_dir, 'input'),
        os.path.join(test_dir, 'label'),
        batch_size = 10)

    class PredCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            pass
            #pred.predict(self.model, eval_img, 'base-%d.png' % epoch, 'out-%d.png' % epoch, False)

    class PSNRCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss = logs['loss'] * 255.
            val_loss = logs['val_loss'] * 255.
            psnr = 20 * math.log10(255. / math.sqrt(loss))
            val_psnr = 20 * math.log10(255. / math.sqrt(val_loss))
            print("\n")
            print("PSNR:%s" % psnr)
            print("PSNR(val):%s" % val_psnr)

    pd_cb = PredCallback()
    ps_cb = PSNRCallback()
    md_cb = ModelCheckpoint(os.path.join(model_dir,'check.h5'), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
    tb_cb = TensorBoard(log_dir=log_dir)

    srcnn_model.fit_generator(
        generator = train_gen,
        steps_per_epoch = 10,
        validation_data = val_gen,
        validation_steps = 10,
        epochs = 1,
        callbacks=[pd_cb, ps_cb, md_cb, tb_cb])

    srcnn_model.save(os.path.join(model_dir,'model.h5'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("model_dir")
    parser.add_argument("train_dir")
    parser.add_argument("test_dir")
    parser.add_argument("--eval_img")
    args = parser.parse_args()
    print(args)

    if not exists(args.model_dir):
        makedirs(args.model_dir)

    train(args.log_dir, args.model_dir, args.train_dir, args.test_dir, args.eval_img)
