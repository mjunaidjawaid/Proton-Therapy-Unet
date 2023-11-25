import Network_elements_instance
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv3D
import Data_Transform
import matplotlib.pyplot as plt
import random

print ('non_empty_patches')

tf.keras.mixed_precision.Policy('float32')

def cartesian_product(*arrays):
    ndim = len(arrays)
    return np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim)

x_z_slice_size = 64
x_z_stride = 32
transformation_list = np.arange(0, 8)   # for data augmentation transformations, 0 to 7 represent different transformations
input_n_channels = 2

Unet_input_shape = (x_z_slice_size, 64, x_z_slice_size)

threshold = 0.0


def Unet_3d(x_in):
    
    block = Network_elements_instance.blocks()

    Multi_Elab_1=block.MultiScaleElaboration(x_in)
    Multi_Elab_1_1=block.MultiScaleElaboration(Multi_Elab_1)    #To be used for concatenation

    Reduction_1=block.Reduction(Multi_Elab_1_1)

    Multi_Elab_2=block.MultiScaleElaboration(Reduction_1)
    Multi_Elab_2_1=block.MultiScaleElaboration(Multi_Elab_2)   #To be used for concatenation

    Reduction_2=block.Reduction(Multi_Elab_2_1)

    Multi_Elab_3=block.MultiScaleElaboration(Reduction_2)
    Multi_Elab_3_1=block.MultiScaleElaboration(Multi_Elab_3)  #To be used for concatenation

    Reduction_3=block.Reduction(Multi_Elab_3_1)

    Multi_Elab_4=block.MultiScaleElaboration(Reduction_3)
    Multi_Elab_4_1=block.MultiScaleElaboration(Multi_Elab_4)

    Expansion_1=block.expansion(Multi_Elab_4_1)

    Concat_1=Concatenate(axis=-1)([Multi_Elab_3_1,Expansion_1])

    Multi_scale_elaboration_5=block.MultiScaleElaboration(Concat_1)
    Multi_scale_elaboration_5_1=block.MultiScaleElaboration(Multi_scale_elaboration_5)

                               
    Expansion_2=block.expansion(Multi_scale_elaboration_5_1)

    Concat_2=Concatenate(axis=-1)([Multi_Elab_2_1,Expansion_2])

    multi_scale_elaboration_6=block.MultiScaleElaboration(Concat_2)
    multi_scale_elaboration_6_1=block.MultiScaleElaboration(multi_scale_elaboration_6)

    Expansion_3=block.expansion(multi_scale_elaboration_6_1)

    Concat_3=Concatenate(axis=-1)([Multi_Elab_1_1,Expansion_3])

    multi_scale_elaboration_7=block.MultiScaleElaboration(Concat_3)
    multi_scale_elaboration_7_1=block.MultiScaleElaboration(multi_scale_elaboration_7)

    Conv=Conv3D (filters=1, 
                 kernel_size=[1,1,1], 
                 strides=[1,1,1],
                 data_format="channels_last",
                 use_bias= True,
                 dtype=np.float32)(multi_scale_elaboration_7_1)
    
    return Conv


input = tf.keras.Input(shape=[64,64,64,2], dtype=np.float32)

Unet_output=Unet_3d(input)

print(np.shape(Unet_output))

model = tf.keras.Model(inputs=input, outputs=Unet_output)

# Data pipeline

transform = Data_Transform.transformation()

class DataGenerator(tf.keras.utils.Sequence):
    'Generate data for keras'

    def __init__(self, path, batch_size, dim, n_channel, slice_size, stride, selection_list,
                 shuffle = True):
        
        self.dim= dim
        self.slice_size= slice_size
        self.stride= stride
        self.batch_size= batch_size
        self.non_empty_list = selection_list #selection list is non_empty list
        self.transformation_list = selection_list
        self.n_channels= n_channel
        self.shuffle= shuffle
        self.path= path
        self.width = dim[1]
        self.on_epoch_end()

    def __len__(self):
        
        elements = np.shape(self.transformation_list)
        return int(np.floor(elements[0] / self.batch_size))
    
    def __getitem__(self,idx):

        'Generate a batch of data'
        indexes= self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        index_list_temp= [self.transformation_list[k] for k in indexes]

        X_in, Y_in = self.__data_generator(index_list_temp)

        print(X_in.shape, Y_in.shape)

        return X_in, Y_in

    def on_epoch_end(self):
        self.indexes= np.arange(len(self.transformation_list))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generator(self, transformation_list):

        X = np.empty((self.batch_size, *self.dim, self.n_channels))   # 7 corresponds to data augmentation
        Y = np.empty((self.batch_size, *self.dim, 1)) 

        for i, ID in enumerate(transformation_list):

            CT_scan = np.load(self.path+'/CT_scans'+'/CT_'+ID[2]+'.npy').astype(np.float32)
            Dose_5K = np.load(self.path+'/Dose_5K'+'/Dose5K_'+ID[2]+'.npy').astype(np.float32)
            Dose_1M = np.load(self.path+'/Dose_1M'+'/Dose1M_'+ID[2]+'.npy').astype(np.float32)

            #Normalization

            CT_scan_norm = (CT_scan + 1000) / 3000
            Dose_5K_norm = ((Dose_5K - np.min(Dose_5K)) / (np.max(Dose_5K) - np.min(Dose_5K)))
            Dose_1M_norm = ((Dose_1M - np.min(Dose_1M)) / (np.max(Dose_1M) - np.min(Dose_1M)))

            threshold = 0.005*np.max(Dose_1M_norm)
            Dose_1M_input = np.where(Dose_1M_norm>=threshold, Dose_1M_norm, 0)

            x_slice = int(ID[0])
            z_slice = int(ID[1])

            CT_scan_slice = CT_scan_norm[x_slice:x_slice+self.slice_size,:, z_slice:z_slice+self.slice_size]
            Dose_5K_slice = Dose_5K_norm[x_slice:x_slice+self.slice_size,:, z_slice:z_slice+self.slice_size]
            Dose_1M_slice = Dose_1M_input[x_slice:x_slice+self.slice_size,:, z_slice:z_slice+self.slice_size]

            transformation_indicator = random.randint(0, 6) # random transformation from 0 to 7
            
            if transformation_indicator == '1':
                CT_scan_slice = transform.flip(CT_scan_slice, 0)
                Dose_5K_slice = transform.flip(Dose_5K_slice, 0)
                Dose_1M_slice = transform.flip(Dose_1M_slice, 0)

            elif transformation_indicator   == '2':
                CT_scan_slice = transform.flip(CT_scan_slice, 1)
                Dose_5K_slice = transform.flip(Dose_5K_slice, 1)
                Dose_1M_slice = transform.flip(Dose_1M_slice, 1)

            elif transformation_indicator == '3':
                CT_scan_slice = transform.flip(CT_scan_slice, 2)
                Dose_5K_slice = transform.flip(Dose_5K_slice, 2)
                Dose_1M_slice = transform.flip(Dose_1M_slice, 2)

            elif transformation_indicator == '4':
                CT_scan_slice = transform.rotation(CT_scan_slice, 1, (0, 2))
                Dose_5K_slice = transform.rotation(Dose_5K_slice, 1, (0, 2))
                Dose_1M_slice = transform.rotation(Dose_1M_slice, 1, (0, 2))

            elif transformation_indicator == '5':
                CT_scan_slice = transform.rotation(CT_scan_slice, 2, (0, 1))
                Dose_5K_slice = transform.rotation(Dose_5K_slice, 2, (0, 1))
                Dose_1M_slice = transform.rotation(Dose_1M_slice, 2, (0, 1))

            elif transformation_indicator == '6':
                CT_scan_slice = transform.rotation(CT_scan_slice, 2, (1, 2))
                Dose_5K_slice = transform.rotation(Dose_5K_slice, 2, (1, 2))
                Dose_1M_slice = transform.rotation(Dose_1M_slice, 2, (1, 2))


            X[i,]= np.stack((Dose_5K_slice, CT_scan_slice), axis=-1)
            Y[i,]= Dose_1M_slice.reshape((*self.dim, 1))

        
        return X, Y

train_path = '/NFSHOME/mspezialetti/sharedFolder/U-net_dataset/training'
validation_path = '/NFSHOME/mspezialetti/sharedFolder/U-net_dataset/validation'

train_batch_size = 12
val_batch_size = 12

training_examples = 785
index_list_train = [f'{num:03}' for num in range (1, training_examples+1)]
index_list_train = np.array(index_list_train)

x_slice_list = np.arange(0, (512 - x_z_slice_size + 1), x_z_stride)
z_slice_list = np.arange(0, (512 - x_z_slice_size + 1), x_z_stride)

#train_resulting_list = cartesian_product(index_list_train, x_slice_list, z_slice_list, transformation_list)

raw_train_non_empty_list = np.load('/NFSHOME/mjawaid/U_net/non_empty_list.npy')

train_non_empty_list = raw_train_non_empty_list[1:]

prams_train = {'selection_list': train_non_empty_list,
                'dim': [64,64,64],
                'n_channel':input_n_channels,
                'slice_size': x_z_slice_size,
                'stride': x_z_stride,
                'path': train_path,
                'batch_size': train_batch_size, # 'batch_size' is not used in the class 'DataGenerator
                'shuffle':True}

training_generator = DataGenerator(**prams_train)

validation_examples = 75
index_list_validation = [f'{num:03}' for num in range (1, validation_examples+1)]
index_list_validation = np.array(index_list_validation)

#val_resulting_list = cartesian_product(index_list_validation, x_slice_list, z_slice_list, transformation_list)

raw_val_non_empty_list = np.load('/NFSHOME/mjawaid/U_net/non_empty_list_validation.npy')
val_non_empty_list = raw_val_non_empty_list[1:]

prams_validation = {'selection_list': val_non_empty_list,
                    'dim': [64,64,64],
                    'n_channel':input_n_channels,
                    'slice_size': x_z_slice_size,
                    'stride': x_z_stride,
                    'path': validation_path,
                    'batch_size': val_batch_size, # 'batch_size' is not used in the class 'DataGenerator
                    'shuffle':True}

validation_generator = DataGenerator(**prams_validation)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-8),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)])     #tf.keras.metrics.Accuracy - does not work

print(model.summary())

history = model.fit(training_generator,
          epochs = 100,
          #verbose = 'auto',
          callbacks= [tf.keras.callbacks.CSVLogger('/NFSHOME/mspezialetti/sharedFolder/3D_Unet/training_log5_oct_csv', separator = ','), 
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                           patience=2, min_lr=0.0000000000001, vferbose=1),
                      tf.keras.callbacks.ModelCheckpoint('/NFSHOME/mspezialetti/sharedFolder/3D_Unet/mymodel_Nov_non_empty_patches[64].keras', 
                                                         monitor='val_loss', 
                                                         verbose=1, 
                                                         save_best_only=True, 
                                                         save_weights_only=False, 
                                                         mode='auto', 
                                                         save_freq='epoch'),
                     tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=10, 
                                                      restore_best_weights=True)],
          validation_data = validation_generator,
          shuffle = True,
          steps_per_epoch = int(np.floor(len(train_non_empty_list) / train_batch_size)/2) ,
          validation_steps=200,
          initial_epoch = 0,
          workers = 30,
          verbose=1) # tensorboard can also be included in callbacks


plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model root mean squared error')
plt.ylabel('root mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("/NFSHOME/mspezialetti/sharedFolder/3D_Unet/Nov_non_empty.jpg")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("/NFSHOME/mspezialetti/sharedFolder/3D_Unet/Nov_non_empty.jpg")


testing_path = '/NFSHOME/mspezialetti/sharedFolder/U-net_dataset/testing'
testing_examples = 89
index_list_test = [f'{num:03}' for num in range (1, testing_examples+1)]
index_list_test = np.array(index_list_test)

model.save('/NFSHOME/mspezialetti/sharedFolder/3D_Unet/mymodel_Nov_non_empty_patches[64].keras')


test_resulting_list = cartesian_product(index_list_test, x_slice_list, z_slice_list, transformation_list)
prams_test = {'selection_list': test_resulting_list,
              'dim': [64,64,64],
              'n_channel':input_n_channels,
              'slice_size': x_z_slice_size,
              'stride': x_z_stride,
              'path': testing_path,
              'batch_size': val_batch_size, # 'batch_size' is not used in the class 'DataGenerator
              'shuffle':True}

test_generator = DataGenerator(**prams_test)

score = model.evaluate(test_generator,
                       workers = 20)
                       
print("The score is:")
print(score)

#model.save('/NFSHOME/mspezialetti/sharedFolder/3D_Unet/mymodel_Oct_LR_no_change.keras')



          
