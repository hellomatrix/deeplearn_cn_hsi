
image_size = 5
pca_components = 10

# first is window size, second is channels
kernel_shape = [[3,pca_components*3],[3,pca_components*3*3]]

fc_out_shape = pca_components*3*2

random_state = 25348
ratio = [6, 2, 2]
batch_size = 100
epoch_times = 10000000

Salinas_origin = 'Salinas_origin'
Salinas = 'Salinas'
data_path = '../hsi_data'

log = './model'