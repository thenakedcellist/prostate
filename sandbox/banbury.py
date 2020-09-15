import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


# %% read dataset from files

# change cwd
os.chdir(r'H:\MSc\project\banbury_data\cornea')

# read data in tsv format
data1 = np.genfromtxt('cornea_eye_1_fp_sub_0__X_20912.1__Y_-1142.71__Time_0__Zdata_4264.95__Zactual_4264.89__Zdifference_-0.0527536__LTSignalUsed_3.txt', delimiter='\t', usecols=(0, 1))

data2 = np.genfromtxt('cornea_eye_1_fp_sub_1__X_20923.1__Y_-1142.71__Time_5__Zdata_4273.41__Zactual_4273.57__Zdifference_0.159993__LTSignalUsed_3.txt', delimiter='\t', usecols=(0, 1))


# %% merge data into single array

if data1[:, 0].all() == data2[:, 0].all():
    data2Neg = np.delete(data2, 0, 1)
    print(data2Neg)
    mergedData = np.append(data1, data2Neg, axis=1)
    print(mergedData)
else:
    print("error")

# this gives mean but of all columns/rows - could possibly use tuple to define, or remove column 0 and replace
avMergedData = np.mean(mergedData, axis=1)


# %% mean centering and normalisation to unit variance

scaledData = preprocessing.scale(mergedData)


# %% plot spectra

plt.figure(1, figsize=(16, 8))
# axes = plt.gca()
# axes.set_xlim(min, max)
# axes.set_ylim(min, max)
plt.plot(data1[:, 1])
plt.figure(2, figsize=(16, 8))
plt.plot(data2[:, 1])
plt.figure(3, figsize=(16, 8))
plt.plot(scaledData[:, 1])
plt.figure(4, figsize=(16, 8))
plt.plot(scaledData[:, 2])


# %% TODO list
'''
TODO
concatenate file strings to animal and acquisition number
create array with:
    column 0: wavenumber
    column 1-n: measured value for wavenumber
create array with:
    column 0 from above
    average of cloumns 1-n fom above
plot this average array
do this for each eye, 11 in total
see from source code how the Banbury group got their average spectra
'''