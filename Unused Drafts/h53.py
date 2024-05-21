import h5py

file_path = r'C:\Users\naman\Desktop\Sign Letter 2\Sign Letter 2\Model\keras_model.h5'
group_key = ('model_weights',)  # Tuple key

with h5py.File(file_path, 'r') as file:
    group_name = group_key[0]  # Extract the group name from the tuple
    group = file[group_name.encode()]  # Convert to byte string and access the group

    # Perform further operations on the group as needed
    print(group.keys())
