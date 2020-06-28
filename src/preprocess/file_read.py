import scipy.io
mat = scipy.io.loadmat('..\\data\\sprites\\sprites_splits.mat')

print(mat["trainidx"][0])

# import h5py
# with h5py.File('..\\data\\sprites\\sprites_0.mat',"r") as f:
# 	# temp = list(f.keys())
# 	# print(temp)
# 	# print(f.keys())
# 	print(f["sprites"].value)
# 	# for key, value in f.items():
# 	# 	print(key,value)


import numpy as np, h5py 

# "D:\Deep-Learning-Assignments\assignment_3\question_2\data\sprites\sprites_splits.mat"
# f = h5py.File('..\\data\\sprites\\sprites_splits.mat','r') 
# print(f.keys())
# masks = f["masks"]
# print(masks[0,0].keys())
# a_group_key = list(f.keys())[3]
# print(a_group_key)
# data = list(f[a_group_key])
# print(len(data))
# print(data)
# print(f[data])
# data = f[masks][0,0]
# data = f.get('masks').value[0][0]
# print(data)
# data = np.array(data)