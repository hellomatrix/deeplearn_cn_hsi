
import tensorflow as tf
from data import data
import config
from hsi_cnn import hsi_cnn



def run_trainning(data_name = None):

    ds = data(data_name)
    data_set = ds.get_train_valid_test()

    train_sets = [data_set[0],data_set[1]]
    valid_sets = [data_set[2],data_set[3]]
    test_sets = [data_set[4],data_set[5]]

    epoch_size = data_set[0].shape[0]
    class_number = ds.class_number

    with tf.Graph().as_default() as gad:

        writer = tf.summary.FileWriter()

        with tf.Session(graph = gad) as sess:

            cnn_nets = hsi_cnn(class_number=class_number)

            for step in range(config.epoch_times*(epoch_size//config.batch_size)):

                cost,_ = cnn_nets.fit(feed_dict = {cnn_nets.})







