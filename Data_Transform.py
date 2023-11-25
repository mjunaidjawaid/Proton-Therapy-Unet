import numpy as np
class transformation:

    def flip(self, x_in, flip_axis):

        flipped= np.flip(x_in, flip_axis)
        return flipped
    
    def rotation(self, x_in, n_rotation, plane):

        rotated=np.rot90(x_in, k = n_rotation, axes=plane)
        return rotated



