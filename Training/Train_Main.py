"""
We were assisted by  https://github.com/qqwweee/keras-yolo3
"""

import os
import sys
import argparse
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from Train_Utils import get_classes, get_anchors, create_model, data_generator_wrapper

def get_parent_dir(n=1):
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

Data_Folder = os.path.join(get_parent_dir(1),'Data')
Image_Folder = os.path.join(Data_Folder,'Source_Video','Training_Frames')
Annotation_file = os.path.join(Image_Folder,'data_train.txt')
Model_Folder = os.path.join(Data_Folder,'Model_Weights')
classname_file = 'data_classes.txt'
anchors_path = 'yolo_anchors.txt'
weights_path = os.path.join(Model_Folder,'yolo.h5')

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--annotation_file", type=str, default=Annotation_file,
        help = "Absolute Path to annotations file (data_train.txt). Default is "+ Annotation_file
        )
    parser.add_argument(
        "--weights_folder_path", type=str, default=None,
        help = "absolute path to the folder to save the trained weights to. Default is None"
        )
    parser.add_argument(
        "--val_split", type=float, default=0.1,
        help = "Percentage of training set to be used for validation. Default is 10%."
        )
    parser.add_argument(
        "--epochs", type=float, default=300,
        help = "Number of epochs for training last layers and number of epochs for fine-tuning layers. Default is 51."
        )
    parser.add_argument(
        "--pre_trained_path", type=str,dest='pre_trained_weights_path', default=None,
        help="Absolute path for pre trained weights. default is None."
    )

    
    FLAGS = parser.parse_args()
    np.random.seed(None)

    weights_folder = FLAGS.weights_folder_path
    class_names = get_classes(classname_file)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (416, 416)
    epoch1, epoch2 = FLAGS.epochs, FLAGS.epochs
    model = create_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path = weights_path)


    if FLAGS.pre_trained_weights_path!=None:
        print('load pre trained weights: ' + FLAGS.pre_trained_weights_path)
        model.load_weights(FLAGS.pre_trained_weights_path)


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.98, patience=3, verbose=1)
    val_split = FLAGS.val_split
    with open(FLAGS.annotation_file) as f:
        lines = f.readlines()

    np.random.shuffle(lines)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=epoch1,
                initial_epoch=0)

        if weights_folder!=None:
            model.save_weights(os.path.join(weights_folder,'trained_weights_stage_1.h5'))


    # Unfreeze and continue training, to fine-tune.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print('Unfreeze all layers.')

        batch_size = 4
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history=model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=epoch1+epoch2,
            initial_epoch=epoch1,
            callbacks=[reduce_lr])

        if weights_folder!=None:
            model.save_weights(os.path.join(weights_folder,'trained_weights_final.h5'))

