import numpy as np

# load npy file
npy_load = np.load('D:/Others/Programming/Project Space/Franklin_Clinton/NPY/Frank_NPY.npy'))

# Show array
npy_load.shape

# Change dimension 3D to 2D
D_npy = npy_load.reshape(800, 2400, order='c')

# Save array to csv file
csv_save = np.savetxt('D:/Others/Programming/Project Space/Franklin_Clinton/CSV/Frank_CSV.csv', D_npy, delimiter=",")

# Load csv file
csv_load = open('D:/Others/Programming/Project Space/Franklin_Clinton/CSV/Frank_CSV.csv', 'r')
# Save to list
csv_list = csv_load.readlines()
# Close csv file
csv_load.close()

# Count csv list
len(csv_list)

csv_list[0]
