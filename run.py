
import tensorflow as tf
from data import data
import config
from hsi_cnn import hsi_cnn
import time
import numpy as np
import os.path


def run_trainning(data_name = None):

    train_ckpt_dir=config.log+'/'+data_name
    train_graph_path = os.path.join(train_ckpt_dir,'graph')
    train_ckpt_path = os.path.join(train_ckpt_dir,'ckpt')

    if not os.path.exists(train_graph_path):
        os.makedirs(train_graph_path)
    if not os.path.exists(train_ckpt_path):
        os.makedirs(train_ckpt_path)



    ds = data(data_name)
    data_set = ds.get_train_valid_test()

    train_sets = [data_set[0],data_set[1]]
    valid_sets = [data_set[2],data_set[3]]
    test_sets = [data_set[4],data_set[5]]

    epoch_size = data_set[0].shape[0]
    class_number = ds.class_number

    with tf.Graph().as_default() as gad:

        writer = tf.summary.FileWriter(train_graph_path)
        with tf.Session(graph = gad) as sess:

            cnn_nets = hsi_cnn(input_dim =ds.img_size_flat ,class_number= ds.class_number)
            saver = tf.train.Saver()

            init = tf.global_variables_initializer()
            sess.run(init)

            for step in range(config.epoch_times*(epoch_size//config.batch_size)):

                start_time = time.time()

                train_x,train_y = ds.next_batch()

                cost,_ = cnn_nets.fit(feed_dict = {cnn_nets.input_x:train_x,cnn_nets.y_true:train_y,cnn_nets.keep_prob:0.5},
                                      sess=sess)

                duration = time.time()-start_time

                if step%100 ==0:
                    print('final model train: step %d,loss=%.5f,time=%.3f sec'%(step,cost,duration))

                    # print('All train data evaluation:')

                    t_summary_loss = sess.run(cnn_nets.train_merged,
                                            feed_dict={cnn_nets.input_x:train_sets[0],
                                                       cnn_nets.y_true:train_sets[1],cnn_nets.keep_prob:1.0})

                    writer.add_summary(t_summary_loss,step)

                    # print('All valid data evaluation:')

                    v_summary_loss = sess.run(cnn_nets.valid_merged,
                                            feed_dict={cnn_nets.input_x:valid_sets[0],
                                                       cnn_nets.y_true:valid_sets[1],cnn_nets.keep_prob:1.0})

                    writer.add_summary(v_summary_loss,step)

                    # print('All test data evaluation:')

                    te_summary_loss = sess.run(cnn_nets.test_merged,
                                            feed_dict={cnn_nets.input_x:test_sets[0],
                                                       cnn_nets.y_true:test_sets[1],cnn_nets.keep_prob:1.0})

                    writer.add_summary(te_summary_loss,step)


                    saver.save(sess,os.path.join(train_ckpt_path,'final_model.ckpt'))




if __name__ == '__main__':

    data_name = config.Salinas
    run_trainning(data_name)


