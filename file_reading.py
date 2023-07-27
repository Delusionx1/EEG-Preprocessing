import numpy as np

f = open('Pre_processing_order_3.txt')
lines = f.readlines()
data_array = lines[-4:]
print(data_array)
arr = []

for i in range(len(data_array)):
    part = np.fromstring(data_array[i][2:-2], dtype=int, sep=' ')
    arr.append(part)

arr = np.asarray(arr)
print(arr, arr.shape)