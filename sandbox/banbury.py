import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


# %% read dataset from files

# change cwd
os.chdir(r'H:\MSc\project\data\banbury\cornea')

# read data in tsv format
data1 = np.genfromtxt('cornea_eye_1_fp_sub_0__X_20912.1__Y_-1142.71__Time_0__Zdata_4264.95__Zactual_4264.89__Zdifference_-0.0527536__LTSignalUsed_3.txt', delimiter='\t', usecols=(0, 1)).T
xdata1 = data1[0:1, :]  # this forms the x data for pyplot and the data to check in each file that wavenumber is the same

data2 = np.genfromtxt('cornea_eye_1_fp_sub_1__X_20923.1__Y_-1142.71__Time_5__Zdata_4273.41__Zactual_4273.57__Zdifference_0.159993__LTSignalUsed_3.txt', delimiter='\t', usecols=(0, 1)).T

# %% merge data into single array
'''
check that the first row matches wavenuber given in first file
IF TRUE
    append return array with second row (y data)
ELSE
    throw mismatch error

return k x n array where:
    k = number of samples
    n = number of y values

'''
mergedData = np.array(data1[1:2, :])  # np array where first row  contains y values of first sample

if xdata1[0:1, :].all() == data2[0:1, :].all():
    data2Neg = np.delete(data2, 0, 0)
    mergedData = np.append(mergedData, data2Neg, axis=0)
else:
    print("error")

# this gives mean but of all columns/rows - could possibly use tuple to define, or remove column 0 and replace
avMergedData = np.mean(mergedData, axis=0)


# %% mean centering and normalisation to unit variance

scaledData = preprocessing.scale(avMergedData)


# %% plot spectra

fig,ax = plt.subplots(1,1)
fig.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
fig.suptitle("Averaged Raman spectra for ", fontsize=16)

xdata = xdata1
ydata = scaledData

ax.plot(xdata, avMergedData)
plt.show()

# TODO create single array with all y values concatenated from the multiple input files
# TODO plot spectra and average spectra for each group (location, animal eye)
# TODO review Banbury source code to see how they got their average spectra
