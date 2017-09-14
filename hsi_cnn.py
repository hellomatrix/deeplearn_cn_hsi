
import tensorflow as tf
import config
import numpy as np

class hsi_cnn(object):

    def __init__(self,input_dim,class_number = None,
                 use_pooling=False,
                 kernel_shape=config.kernel_shape,
                 image_size = config.image_size,
                 fc_out_shape = config.fc_out_shape,
                 input_channels=config.pca_components,
                 optimizer = tf.train.AdamOptimizer()
                 ):

        # input image is a vector
        self.img_size_flat = input_dim

        self.kernel_shape = kernel_shape
        self.image_size = image_size
        self.input_channels=input_channels
        self.use_pooling = use_pooling
        self.class_number = class_number
        self.fc_out_shape = fc_out_shape
        self.keep_prob = tf.placeholder(tf.float32)

        self.input_x = tf.placeholder(tf.float32,shape=[None,self.img_size_flat])
        self.input_y = tf.placeholder(tf.float32,shape=[None])

        self.x_img = tf.reshape(self.input_x,[-1,self.image_size,self.image_size,self.input_channels])

        # self.input_x_image = tf.reshape(self.input_x,[-1,self.image_size,self.image_size,self.num_channels])
        # self.input_x_image = tf.placeholder(tf.float32,[None,self.image_size,self.image_size,self.class_number])


        self.y_true = tf.placeholder(tf.float32,shape =[None,self.class_number])
        self.y_true_class = tf.arg_max(self.y_true,dimension=1) # arg_max along axis 1 (column)

        with tf.variable_scope('layer_conv1'):
            layer_conv1, weigths_conv1 =\
                self.new_conv_layer(input = self.x_img,
                               filter_size = self.kernel_shape[0][0],
                               input_channels=self.input_channels,
                               output_channels = self.kernel_shape[0][1],
                               use_pooling = self.use_pooling
                               )

        with tf.variable_scope('layer_conv2'):
            layer_conv2, weigths_conv2 =\
                self.new_conv_layer(input = layer_conv1,
                               filter_size = self.kernel_shape[1][0],
                               input_channels=self.kernel_shape[0][1],
                               output_channels = self.kernel_shape[1][1],
                               use_pooling = self.use_pooling
                               )

            # flatten the last layer of nets
            layer_flatten, dim_feature = self.flatten_layer(layer_conv2)

        # full connected the flatten layer to out put
        with tf.variable_scope('layer_fc'):
            layer_fc,_ = self.new_fc_layer(input = layer_flatten,input_dim = dim_feature,
                                         output_dim=self.fc_out_shape, softplus=True)


            layer_fc_dropout = tf.nn.dropout(layer_fc,self.keep_prob)

            layer_out,_ = self.new_fc_layer(input = layer_fc_dropout,input_dim = self.fc_out_shape,
                                         output_dim=self.class_number, softplus=False)

        # softmax the output layer
        y_pred = tf.nn.softmax(layer_out)
        # the result is a vector , the index of biggest scaler in vector is the class number
        y_pred_cls = tf.arg_max(y_pred,dimension=1)

        # cost
        # logits are matrix:
        # row: the size of batch
        # column: the softmax vector(the prabability),
        # labels are matrix:
        # row: the size of batch
        # column : the one-hot form of labels.

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = self.y_true,logits = layer_out)
        self.cost =tf.reduce_mean(cross_entropy)

        # optimizer
        self.optimizer = optimizer.minimize(self.cost)
        correct_prediction = tf.equal(y_pred_cls,self.y_true_class)
        self.accuarcy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        self.train_summary=[]
        self.valid_summary=[]
        self.test_summary=[]

        ## precision
        with tf.name_scope('train_summary') as train_summary:
            loss = tf.summary.scalar('loss',self.cost)
            accuarcy =  tf.summary.scalar('accuarcy',self.accuarcy)
            self.train_summary.append(loss)
            self.train_summary.append(accuarcy)
            self.train_merged=tf.summary.merge(self.train_summary,train_summary)

        ## precision valid
        with tf.name_scope('valid_summary') as valid_summary:
            loss = tf.summary.scalar('loss', self.cost)
            accuarcy = tf.summary.scalar('accuarcy', self.accuarcy)
            self.valid_summary.append(loss)
            self.valid_summary.append(accuarcy)
            self.valid_merged = tf.summary.merge(self.valid_summary, valid_summary)

        ## precision test
        with tf.name_scope('test_summary') as test_summary:
            loss = tf.summary.scalar('loss', self.cost)
            accuarcy = tf.summary.scalar('accuarcy', self.accuarcy)
            self.test_summary.append(loss)
            self.test_summary.append(accuarcy)
            self.test_merged = tf.summary.merge(self.test_summary, test_summary)

        #train
    def fit(self,feed_dict,sess):
        return sess.run((self.cost,self.optimizer),feed_dict = feed_dict)


    def new_conv_layer(self,input,filter_size,input_channels,output_channels,use_pooling=False):

        # do not use input as parameters cause tensor is not a number
        shape = [filter_size,filter_size,input_channels,output_channels]
        weights = self.new_weights(shape=shape)
        bias = self.new_bias(length=output_channels)
        layer = tf.nn.conv2d(input = input, filter=weights,strides=[1,1,1,1],padding='VALID')
        layer += bias

        if use_pooling:

            layer=tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

        layer=tf.nn.relu(layer)

        return layer,weights

    def plot_images(self):

        return

    def new_weights(self,shape):
        return tf.Variable(tf.truncated_normal(shape,stddev=0.5))

    def new_bias(self,length):
        return tf.Variable(tf.constant(0.05,shape=[length]))

    def flatten_layer(self,layer):

        shape = layer.get_shape()

        dim_feature = np.int(shape[1]*shape[2]*shape[3])

        layer_flat = tf.reshape(layer,[-1,dim_feature])

        return layer_flat,dim_feature

    def new_fc_layer(self,input,input_dim = None, output_dim = None, softplus=False):

        weights = self.new_weights(shape = [input_dim,output_dim])
        biases = self.new_bias(length = output_dim)

        layer = tf.add(tf.matmul(input,weights),biases)

        if softplus:
            layer = tf.nn.softplus(layer)

        return layer,weights



   # if __name__=='__main__':

        # from hsi_cnn import hsi_cnn
        #
        # with tf.Graph().as_default() as gad:
        #
        #     with tf.Session() as sess:
        #
        #         cnn_net = hsi_cnn()










































