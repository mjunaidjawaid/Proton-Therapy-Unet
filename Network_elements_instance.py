import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, UpSampling3D, Conv3DTranspose
from tensorflow.keras.layers import  LeakyReLU, Concatenate,BatchNormalization
from tensorflow.keras.layers import AveragePooling3D, MaxPooling3D
import tensorflow_addons as tfa

class blocks:

    def MultiScaleElaboration(self, x_input): #input is the 3d image with 2 channels. First channel is the CT-Scan, and the second MCTopas outpit. Type: numpy array
                                              #channel last data format
        Kernel_size_1= [1,1,1]
        stride_1= [1,1,1]

        conv_1=Conv3D (filters=12, 
                       kernel_size=Kernel_size_1, 
                       strides=stride_1,
                       data_format="channels_last",
                       use_bias= True)(x_input)
        
        conv_1_norm = tfa.layers.InstanceNormalization()(conv_1)
        
        conv_1_NL= LeakyReLU(0.3)(conv_1_norm)

        Kernel_size_2=[3,3,3]
        stride_2=[1,1,1]

        conv_2=Conv3D (filters=12,
                       kernel_size=Kernel_size_2,
                       strides= stride_2,
                       padding="same",
                       data_format="channels_last",
                       use_bias=True)(x_input)

        conv_2_norm = tfa.layers.InstanceNormalization()(conv_2)
        
        
        conv_2_NL=LeakyReLU(0.3)(conv_2_norm)
                        
        
        conv_3=Conv3D(filters=12,
                      kernel_size=Kernel_size_2,
                      strides=stride_2,
                      padding="same",
                      data_format="channels_last",
                      use_bias=True)(x_input)
        
        conv_3_norm = tfa.layers.InstanceNormalization()(conv_3)
        
        conv_3_NL=LeakyReLU(0.3)(conv_3_norm)
        
        conv_3_1=Conv3D(filters=12,
                        kernel_size=Kernel_size_2,
                        strides=stride_2,
                        padding="same",
                        data_format="channels_last",
                        use_bias=True)(conv_3_NL)
        
        conv_3_1_norm = tfa.layers.InstanceNormalization()(conv_3_1)
        
        conv_3_1_NL=LeakyReLU(0.3)(conv_3_1_norm)
        
        conv_4=Conv3D(filters=12,
                      kernel_size=Kernel_size_2,
                      strides=stride_2,
                      padding="same",
                      data_format="channels_last",
                      use_bias=True)(x_input)
        
        conv_4_norm = tfa.layers.InstanceNormalization()(conv_4)
        
        
        conv_4_NL=LeakyReLU(0.3)(conv_4_norm)


        conv_4_1=Conv3D(filters=12,
                        kernel_size=Kernel_size_2,
                        strides=stride_2,
                        padding="same",
                        data_format="channels_last",
                        use_bias=True)(conv_4_NL)
        
        conv_4_1_norm = tfa.layers.InstanceNormalization()(conv_4_1)

        
        conv_4_1_NL=LeakyReLU(0.3)(conv_4_1_norm)
                        
        
        conv_4_2=Conv3D(filters=12,
                        kernel_size=Kernel_size_2,
                        strides=stride_2,
                        padding="same",
                        data_format="channels_last",
                        use_bias=True)(conv_4_1_NL)
        
        conv_4_2_norm = tfa.layers.InstanceNormalization()(conv_4_2)
        
        conv_4_2_NL=LeakyReLU(0.3)(conv_4_2_norm)
        
        Concat= Concatenate(axis=-1)([conv_1_NL, conv_2_NL, conv_3_1_NL, conv_4_2_NL])
        
        conv_5=Conv3D(filters=24,
                      kernel_size=Kernel_size_1,
                      strides=stride_1,
                      padding="same",
                      data_format="channels_last",
                      use_bias=True)(Concat)

        return Concatenate(axis=-1)([x_input, conv_5])
    

    def Reduction(self, x_in):

        max_pool=MaxPooling3D(pool_size=(2,2,2),
                              strides=(2,2,2),
                              padding="valid")(x_in)
        
        avg_pool=AveragePooling3D(pool_size=(2,2,2),
                                  strides=(2,2,2,),
                                  padding="valid")(x_in)
        
        conv=Conv3D(filters=8,
                    kernel_size=(2,2,2),
                    strides=(2,2,2),
                    padding="valid",
                    data_format="channels_last",
                    use_bias=True)(x_in)
        
        conv_norm = tfa.layers.InstanceNormalization()(conv)
        
        conv_NL=LeakyReLU(0.3)(conv_norm)
        
        concat=Concatenate(axis=-1)([max_pool,avg_pool,conv_NL])

        filter=np.shape(x_in)

        conv_1=Conv3D(filters=filter[-1],
                    kernel_size=(1,1,1),
                    strides=(1,1,1),
                    padding="valid",
                    data_format="channels_last",
                    use_bias=True)(concat)
        
        conv_1_norm = tfa.layers.InstanceNormalization()(conv_1)

        
        conv_1_NL=LeakyReLU(0.3)(conv_1_norm)
        
        return conv_1_NL
    
    def expansion(self, input):    

        input_shape=input.shape
        #print(input_shape)
        channel=int(input_shape[-1]/2)

        conv=Conv3D(filters=channel,
                    kernel_size=(1,1,1),
                    strides=(1,1,1),
                    padding="valid",
                    data_format="channels_last",
                    use_bias=True)(input)
        
        conv_norm = tfa.layers.InstanceNormalization()(conv)
        
        
        conv_NL=LeakyReLU(0.3)(conv_norm)

        up_sample=UpSampling3D(size=(2,2,2))(conv_NL)

        conv_transpose=Conv3DTranspose(filters=channel,
                                       kernel_size=(3,3,3),
                                       strides=(2,2,2),
                                       padding='same',
                                       data_format='channels_last',
                                       use_bias=True)(conv_NL)
        
        concat=Concatenate(axis=-1)([up_sample, conv_transpose])

        #shape=tf.shape(input)

        conv_1=Conv3D(filters=channel,
                    kernel_size=(1,1,1),
                    strides=(1,1,1),
                    padding="valid",
                    data_format="channels_last",
                    use_bias=True)(concat)
        
        conv_1_norm = tfa.layers.InstanceNormalization()(conv_1)
        
        conv_1_NL=LeakyReLU(0.3)(conv_1_norm)

        #print(conv_1_NL.shape)

        return conv_1_NL
    


