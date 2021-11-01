import os
import time
import tensorflow as tf
from keras import callbacks

class HelperFuncs:
    def __init__(self):
        print("constructor of HelperFuncs")

    def get_run_logdir(self, path):
        root_logdir = os.path.join(os.curdir, path)
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)

    # Borrowed from: https://www.geeksforgeeks.org/choose-optimal-number-of-epochs-to-train-a-neural-network-in-keras/
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                            mode ="min", patience = 5, 
                                            restore_best_weights = True)

    def exponential_decay(self, lr0, s):
        def exponential_decay_fn(epoch):
            exp = lr0 * 0.1**(epoch / s)
            tf.summary.scalar('learning rate', data=exp, step=epoch)
            return exp
        return exponential_decay_fn

    exponential_decay_fn = exponential_decay(lr0=0.01, s=10)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
