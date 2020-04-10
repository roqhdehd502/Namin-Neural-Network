import numpy as np
import PIL.Image as pilimg
from PIL import Image

# Read image
im = pilimg.open('{YOUR IMAGE FILE PATH}')

# Display image
im.show()

# Fetch image pixel data to numpy array
pix = np.array(im)

# Save the array to npy file
npy_save = np.save('{YOUR NPY FILE PATH}', pix)

# Load the npy file
npy_load = np.load('{YOUR NPY FILE PATH}.npy')

# Show array
npy_load
