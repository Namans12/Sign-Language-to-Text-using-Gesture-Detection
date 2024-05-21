import h5py

# Open the H5 file
file_path = 'C:/Users/naman/Desktop/Sign Letter 2/Sign Letter 2/Model/keras_model.h5'
file = h5py.File(file_path, 'r')

# List all the top-level objects in the file
print(list(file.keys()))

# Access a specific dataset in the file
dataset = file['model_weights']

# Read the data from the dataset
data = dataset[()]

# Close the file when you're done
file.close()