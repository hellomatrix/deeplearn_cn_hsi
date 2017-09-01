
import numpy as np
import config
import scipy.io as sio
from sklearn.decomposition import PCA
import tensorflow as tf

class data(object):

    def __init__(self,data_name = None,
                 ratio = config.ratio,
                 random_state = config.random_state,
                 batch_size = config.batch_size,
                 pca_componets = config.pca_components,
                 image_size = config.image_size
                 ):

        self.hsi_file = config.data_path + '/' + data_name + '/' + data_name + '.mat'
        self.gnd_file = config.data_path + '/' + data_name + '/' + data_name + '_gt.mat'

        hi = sio.loadmat(self.hsi_file)
        gi = sio.loadmat(self.gnd_file)

        hsi_img = hi[list(hi.keys())[-1]]
        # gnd_img = gi[list(gi.keys())[-1]] #uint8 array seems like image, so  0-1 = 255 !!!!!
        gnd_img = np.int32(gi[list(gi.keys())[-1]]) # !!! be careful about the data type

        print('img_gt shape:[%d,%d],img shape:[%d,%d,%d]'%(gnd_img.shape[0],gnd_img.shape[1],
                                                   hsi_img.shape[0],hsi_img.shape[1],hsi_img.shape[2]))

        self.idx_in_epoch = 0
        self.epoch_num = 0
        self.image_size = image_size

        self.pca_components = pca_componets
        self.batch_size = batch_size
        self.img_2d_shape = gnd_img.shape
        self.ratio = ratio
        self.class_number = np.max(np.max(gnd_img))
        self.random_state = random_state

        print('label min = %d, label max = %d, class_number = %d'
              %(np.min(np.min(gnd_img)),np.max(np.max(gnd_img)),self.class_number))
        for i in range(self.class_number+1):
            print('lable %d pixels of origin: %d'%(i,gnd_img[gnd_img == i].shape[0]))

        # # Scales all values in the ndarray ndar to be between 0 and 1
        self.hsi_img = self.scale_to_unit_interval(hsi_img)

        # class index from 0 - 15, good for softmax
        self.gnd_img = gnd_img - 1
    ## for testing ------------------------------------
        print('label min = %d, label max = %d'
              % (np.min(np.min(self.gnd_img)), np.max(np.max(self.gnd_img))))

        for i in range(self.class_number + 1):
            print('lable %d pixels after changing: %d' % (i - 1, self.gnd_img[self.gnd_img == i - 1].shape[0]))

    ## for testing ------------------------------------

        self.split_mask = self.get_split_mask()
        # remove unlabel pixels
        self.split_mask[self.gnd_img == -1] = '-1'
        temp = self.split_mask[self.split_mask == '-1']
        print('unlable pixels: %d'%(temp.shape[0]))

        self.data_sets = self.get_train_valid_test()
        self.train_examples_number = self.data_sets[0].shape[0]


    def scale_to_unit_interval(self, ndar, eps=1e-8):
        """ Scales all values in the ndarray ndar to be between 0 and 1 """
        ndar = np.float64(ndar.copy())
        ndar -= ndar.min()
        ndar *= 1.0 / (ndar.max() + eps)
        return ndar


    def get_split_mask(self):

        ratio = self.ratio
        random_state = self.random_state

        rand_num_generator = np.random.RandomState(random_state)
        random_mask = rand_num_generator.random_integers(1, sum(ratio), self.img_2d_shape)
        split_mask = np.array([['tests'] * self.img_2d_shape[1], ] * self.img_2d_shape[0])
        split_mask[random_mask <= ratio[0]] = 'train'
        split_mask[(random_mask <= ratio[1] + ratio[0]) * (random_mask > ratio[0])] = 'valid'

        print('split_mask.shape:[%d,%d]'%(split_mask.shape[0],split_mask.shape[1]))

        return split_mask


    def get_train_valid_test(self):

        split_mask = self.split_mask
        batch_size = self.batch_size
        pca_components = self.pca_components
        img_size = self.image_size

        ##-------------
        print('train data shape before pca:[%d,%d,%d]'
              %(self.hsi_img.shape[0],self.hsi_img.shape[1],self.hsi_img.shape[2]))
        ##-------------

        # PCA the data
        pca = PCA( n_components = pca_components)
        pca_hsi_img = pca.fit_transform(np.reshape(self.hsi_img, [-1, self.hsi_img.shape[2]]))
        pca_hsi_img = np.reshape(pca_hsi_img,[self.hsi_img.shape[0],self.hsi_img.shape[1],-1])


        #construct the img patches
        r = self.image_size//2
        print('half of the window:',r)

        new_pca_hsi_img = np.zeros([split_mask.shape[0],
                                    split_mask.shape[1],
                                    img_size*img_size*pca_components])

        for i in range(split_mask.shape[0]-2*r):
            for j in range(split_mask.shape[1]-2*r):
                 new_pca_hsi_img[i+r,j+r,:] = np.reshape(pca_hsi_img[i:i+2*r+1,j:j+2*r+1],[-1,])



        #cut edge
        split_mask = split_mask[2:split_mask.shape[0]-2,2:split_mask.shape[1]-2]
        gnd_img = self.gnd_img[2:self.gnd_img.shape[0]-2,2:self.gnd_img.shape[1]-2]
        new_pca_hsi_img = new_pca_hsi_img[2:new_pca_hsi_img.shape[0]-2,2:new_pca_hsi_img.shape[1]-2]

        print('origin hsi image shape:',pca_hsi_img.shape)
        print('new patches image :',new_pca_hsi_img.shape)
        print('new mask image :',split_mask.shape)
        print('new grandtruth image :',gnd_img.shape)

        train_data_x = new_pca_hsi_img[split_mask=='train']
        train_data_y = gnd_img[split_mask=='train']

        ##-------------
        print('train data shape after pca:',train_data_x.shape)
        print('train data labels after pca:',train_data_y.shape)
        ##-------------

        valid_data_x = new_pca_hsi_img[split_mask=='valid']
        valid_data_y = gnd_img[split_mask=='valid']

        test_data_x = new_pca_hsi_img[split_mask=='tests']
        test_data_y = gnd_img[split_mask=='tests']

        print('origin train pixels :%d, origin valid pixels:%d, origin tests pixels:%d'\
              % (train_data_x.shape[0], valid_data_x.shape[0], test_data_x.shape[0]))

        # tackle the batch size mismatch problem
        mis_match = train_data_x.shape[0] % batch_size
        if mis_match != 0:
            mis_match = batch_size - mis_match

            train_data_x = np.vstack((train_data_x, train_data_x[0:mis_match, :]))
            train_data_y = np.hstack((train_data_y, train_data_y[0:mis_match]))

        mis_match = valid_data_x.shape[0] % batch_size
        if mis_match != 0:
            mis_match = batch_size - mis_match
            valid_data_x = np.vstack((valid_data_x, valid_data_x[0:mis_match, :]))
            valid_data_y = np.hstack((valid_data_y, valid_data_y[0:mis_match]))

        mis_match = test_data_x.shape[0] % batch_size
        if mis_match != 0:
            mis_match = batch_size - mis_match
            test_data_x = np.vstack((test_data_x, test_data_x[0:mis_match, :]))
            test_data_y = np.hstack((test_data_y, test_data_y[0:mis_match]))

        print('modified train pixels:%d, modified valid pixels:%d, modified tests pixels:%d'\
              % (train_data_x.shape[0], valid_data_x.shape[0], test_data_x.shape[0]))



        # modify the data to 4D tensor, labels to one-hot labels

        print('origin train label index one: %d'%train_data_y[0])
        train_data_y = tf.one_hot(
            train_data_y,
            self.class_number,
            on_value=None,
            off_value=None,
            axis=-1,
            dtype=None,
            name=None
            )
        print('one hot train label index one: ',(tf.Session().run(train_data_y[0])))
        print('train_set labels shape:',train_data_y.shape)

        return [train_data_x,train_data_y,valid_data_x,valid_data_y,test_data_x,test_data_y]

    def next_batch(self):
        start = self.idx_in_epoch
        self.idx_in_epoch +=self.batch_size

        if self.idx_in_epoch >self.train_examples_number:

            self.epoch_num = self.epoch_num+1

            perm = np.array(self.train_examples_number)
            np.random.shuffle(perm)
            self.data_sets[0]=self.data_sets[0][perm]
            self.data_sets[1]=self.data_sets[1][perm]

            start = 0

            self.idx_in_epoch = self.batch_size
            assert self.idx_in_epoch<=self.batch_size

        end = self.idx_in_epoch

        print('start idx = %d, end idx = %d'%(start,end))

        return self.data_sets[0][start:end,:],self.data_sets[1][start:end]


    if __name__ == '__main__':

        from data import data

        data_test = data(config.Salinas)

        # print(data_test[0].shape, data_test[1].shape, data_test[2].shape, data_test[3].shape, data_test[4].shape,
        #       data_test[5].shape)

        for i in range(2):
            train_x,train_y = data_test.next_batch()
            print('test next batch: train_x and train_y')
            print(train_x[0:3,0:3],train_y[0:3])










