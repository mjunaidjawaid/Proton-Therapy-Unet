import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

model_path = '/NFSHOME/mspezialetti/sharedFolder/3D_Unet/mymodel_Nov_non_empty_patches[64].keras'

# Define InstanceNormalization as a custom layer
class InstanceNormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InstanceNormalizationLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tfa.layers.InstanceNormalization()(inputs)

# Load your saved model with custom layers
model = tf.keras.models.load_model(model_path, custom_objects={'InstanceNormalizationLayer': InstanceNormalizationLayer})
print('Model loaded successfully')


# Define the shapes and stride
input_shape = (1, 512, 64, 512,2)
window_size = (1, 64, 64, 64, 2)
stride = (32, 0, 32)

output_shape = (1,512,64,512,1)

CT_scan = np.load('/NFSHOME/mspezialetti/sharedFolder/U-net_dataset/training/CT_scans/CT_010.npy', allow_pickle=True).astype(np.float32)
Dose_5K = np.load('/NFSHOME/mspezialetti/sharedFolder/U-net_dataset/training/Dose_5K/Dose5K_010.npy', allow_pickle=True).astype(np.float32)
Dose_ref = np.load('/NFSHOME/mspezialetti/sharedFolder/U-net_dataset/training/Dose_1M/Dose1M_010.npy', allow_pickle=True).astype(np.float32)

dummy = np.zeros((512,64,512), dtype=np.float32)

input = np.stack((CT_scan, Dose_5K), axis=-1)

input = np.reshape(input, input_shape)

'''
dummy_1 = input[0,:,32,:,0]
dummy_2 = input[0,:,32,:,1]

plt.imsave('/NFSHOME/mspezialetti/sharedFolder/plots/CT_070.jpg', dummy_1)
plt.imsave('/NFSHOME/mspezialetti/sharedFolder/plots/Dose_5k_070.jpg', dummy_2)
'''


mask = np.zeros(output_shape, dtype=np.int32)

output = np.zeros(output_shape, dtype=np.float32)

window = np.zeros(window_size, dtype=np.float32)

for x in range(0, input_shape[1]-stride[0], stride[0]):
    for z in range(0, input_shape[1]-stride[2], stride[2]):
            # Define the window limits
        x_start = x
        x_end = x + window_size[1]
        print(x_start, x_end)
        z_start = z
        z_end = z + window_size[3]

            # Extract the window from the input array
        window = input[:,x_start:x_end, :, z_start:z_end,:]

        window = np.reshape(window, window_size)

        print(window.shape)

            # Update the corresponding region in the output array and mask
        output[:,x_start:x_end, :, z_start:z_end,:] += model.predict(window)
        mask[:,x_start:x_end, :, z_start:z_end,:] += 1

# Divide the accumulated values by the mask to get the v
output /= mask


np.save('/NFSHOME/mjawaid/U_net/pred_test_010_[64].npy', output)
print('Prediction saved successfully')




