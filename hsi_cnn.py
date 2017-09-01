
import tensorflow as tf
import config

class hsi_cnn(object):

    def __init__(self,input_dim,class_number = None,
                 kernel_shape=config.kernel_shape,
                 image_size = config.image_size,
                 input_num_channels=config.pca_components,
                 use_pooling = True,
                 optimizer = tf.train.AdamOptimizer()
                 ):

        self.input_dim = input_dim

        self.kernel_shape = kernel_shape
        self.image_size = image_size
        self.input_num_channels=input_num_channels
        self.use_pooling = use_pooling
        self.class_number = class_number

        self.input_x = tf.placeholder(tf.float32,[None,self.input_dim])
        self.input_y = tf.placeholder(tf.float32,[None])

        # self.input_x_image = tf.reshape(self.input_x,[-1,self.image_size,self.image_size,self.num_channels])

        self.input_x_image = tf.placeholder(tf.float32,[None,self.image_size,self.image_size,self.class_number])

        self.y_true = tf.placeholder(tf.float32,[None,self.class_number],name='y_true')
        self.y_true_class = tf.arg_max(self.y_true,dimension=1) # arg_max along axis 1 (column)

        with tf.VariableScope('layer_conv1'):
            layer_conv1, weigths_conv1 =\
                self.new_conv_layer(input = self.input_x_image,
                               num_input_channels = self.input_num_channels,
                               filter_size = self.kernel_shape[0][0],
                               num_flters = self.kernel_shape[0][1],
                               use_pooling = self.use_pooling
                               )

        with tf.VariableScope('layer_conv2'):
            layer_conv2, weigths_conv2 =\
                self.new_conv_layer(input = layer_conv1,
                               num_input_channels = self.kernel_shape[0][0],
                               filter_size = self.kernel_shape[1][0],
                               num_flters = self.kernel_shape[1][1],
                               use_pooling = self.use_pooling
                               )

            layer_flatten, dim_feature = self.flatten_layer(layer_conv2)

        with tf.VariableScope('layer_fc'):

            layer_fc = self.new_fc_layer(input = layer_flatten,input_dim = dim_feature,
                                         output_dim=self.class_numer, use_relu=True)

        # softmax the output layer
        y_pred = tf.nn.softmax(layer_fc)
        # the result is a vector , the index of biggest scaler in vector is the class number
        y_pred_cls = tf.arg_max(y_pred,dimension=1)

        # cost
        # logits are matrix:
        # row: the size of batch
        # column: the softmax vector(the prabability),

        # labels are matrix:
        # row: the size of batch
        # column : the one-hot form of labels.
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = self.y_true,logits = layer_fc)
        self.cost =tf.reduce_mean(cross_entropy)

        # optimaizer
        self.optimizer = optimizer.minimize(self.cost)

        correct_prediction = tf.equal(y_pred_cls,self.y_true_class)

        accuarcy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


        #train
    def fit(self,feed_dict,sess=None):
        return sess.run([self.cost,self.optimizer],feed_dict = feed_dict)



    def new_conv_layer(self,input,num_input_channels,filter_size,num_out_channels,use_pooling=False):

        shape = [filter_size,filter_size,num_input_channels,num_out_channels]
        weights = self.new_weights(shape=shape)
        bias = self.new_bias(length=num_out_channels)
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

        dim_feature = shape[1:4].num_elements()# ??

        layer_flat = tf.reshape(layer,[-1,dim_feature])

        return layer_flat

    def new_fc_layer(self,input,input_dim = None, output_dim = None, use_relu=False):

        weights = self.new_weights(shape = [input_dim,output_dim])
        biases = self.new_bias(length = output_dim)

        layer = tf.add(tf.multiply(input,weights),biases)

        if use_relu:
            layer = tf.nn.relu(layer)

        return layer,weights



    if __name__=='__main__':

        # from hsi_cnn import hsi_cnn
        #
        # with tf.Graph().as_default() as gad:
        #
        #     with tf.Session() as sess:
        #
        #         cnn_net = hsi_cnn()










































