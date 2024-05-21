import h5py

file_path = r'C:\Users\naman\Desktop\Sign Letter 2\Sign Letter 2\Model\keras_model.h5'

with h5py.File(file_path, 'r') as file:
    print(list(file.keys()))