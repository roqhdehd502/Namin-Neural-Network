import os, shutil, keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


base_dir = '{YOUR DIRECTORY}'
img_dir = '{YOUR IMG DIRECTORY}'


# Count images
len(os.listdir(img_dir))


# Check 10 images in folder
os.listdir(img_dir)[:10]


# Set directory copied images
copyImg_dir = os.path.join(base_dir, 'copyImg')


# Make a directory
os.mkdir(copyImg_dir)


# Copy 10 Franklin images to copyImg_dir
fnames = ['Frank{}.jpg'.format(i) for i in range(10)]

for fname in fnames:
    src = os.path.join(img_dir, fname)
    dst = os.path.join(copyImg_dir, fname)
    shutil.copyfile(src, dst)
    

# Check it copied images
os.listdir(copyImg_dir)


# UDF(User Definition Function) of preprocessing image into a 4D tensor
def preprocess_img(img_path, target_size = 100):
    from keras.preprocessing import image
    
    img = image.load_img(img_path, target_size=(target_size, target_size))
    img_tensor = image.img_to_array(img)
    
    # Expand a dimension
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    # Scaling into [0, 1]
    img_tensor /= 255
    
    return img_tensor
    
    
# Layout (Set the correct number of pictures.)
n_pic = 10
n_col = 5
n_row = int(np.ceil(n_pic / n_col)) # Row: 2


# Plot & Margin(Image boundary) size
target_size = 100
margin = 3


# Blank matrix to store results
total = np.zeros((n_row * target_size + (n_row - 1) * margin, n_col * target_size + (n_col - 1) * margin, 3))


# Append the image tensors to the 'total matrix'
img_seq = 0

for i in range(n_row):
    for j in range(n_col):
        fname = 'Frank{}.jpg'.format(img_seq)
        img_path = os.path.join(copyImg_dir, fname)
        
        img_tensor = preprocess_img(img_path, target_size)
        
        horizontal_start = i * target_size + i * margin
        horizontal_end = horizontal_start + target_size
        vertical_start = j * target_size + j * margin
        vertical_end = vertical_start + target_size
        
        total[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = img_tensor[0]
        
        img_seq += 1
        
        
# Display the images in grid
plt.figure(figsize=(200, 200))
plt.imshow(total)
plt.show()
