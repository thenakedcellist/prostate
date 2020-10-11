'''

----------------------------------------------------

Directory notes
This directory contains data from the Banbury directories
converted to an architecture my program can use.
Three subdirectories and a data file exist:
/by_tissue_by_animal
    each csv file contains all spectra obtained from a single tissue from a single animal
/all_tissues_by_animal
    each csv file contains all spectra obtained from all tissues from a single animal
/by_tissue_all_animals
    each csv file contains all spectra obtained from a single tissue from all animals
all_data.csv
    a single file which contains all spectra obtained from all tissues from all animals

Column 1 in each file contains a systematic unique label describing that sample and its hierarchy
    animal_tissue_sample
        animal type: int
            “pig” + index integer of animal
        tissue type: str
            “cornea”
            “lens”
            “optic_nerve”
            “retina”
            “vitreous”
        sample type: int
            “sample” + index integer of sample obtained

----------------------------------------------------

each file of the banbury data contains two columns:
    0: wavenumber
    1: signal intensity


samples exist in a data hierarchy
    animal
        tissue
            sample

this programme aims to extract the data from each individual file
and to convert it into output files with the following row structure:
    0: wavenumber
    1-n: signal intensity

method_list
cwd to banbury data
search directory for subdirectories, and files within
open all files for unique animal_tissue
    set row 0 of sample 0 as row 0 of output_data
    compare row 0 of file k with row 0 of output_data
        IF same
            appended row 1 of file k to outpit_data
        ELSE
            throw error "wavenumbers do not match"
save csv file in directory matching file name animal_tissue
Combine animal_tissue files into single animal and tissue files and place in relevant directories
Combine animal files into single data file and place im root

review Banbury source code to see how they got their average spectra
define functions to plot spectra and average spectra
    method for single spectrum
    method for average spectrum

'''

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root_dir = Path('/Users/dan/Documents/UoY/MSc/project')

def change_cwd(tissue):
   '''change working directory to tissue directory'''
   if tissue not in ("cornea", "lens", "optic_nerve", "retina", "vitreous"):
      raise Exception("invalid tissue type")
   else:
      tissue_dir = Path(root_dir / 'data' / 'banbury' / tissue)
      os.chdir(tissue_dir)
      print(Path.cwd())

def cwd_to_root():
   os.chdir(root_dir)
   print(Path.cwd())


'''

# %% read dataset from files

# change cwd
os.chdir(r'H:\MSc\project\data\banbury\cornea')

# read data in tsv format
data1 = np.genfromtxt('cornea_eye_1_fp_sub_0__X_20912.1__Y_-1142.71__Time_0__Zdata_4264.95__Zactual_4264.89__Zdifference_-0.0527536__LTSignalUsed_3.txt', delimiter='\t', usecols=(0, 1)).T
xdata1 = data1[0:1, :]  # this forms the x data for pyplot and the data to check in each file that wavenumber is the same

data2 = np.genfromtxt('cornea_eye_1_fp_sub_1__X_20923.1__Y_-1142.71__Time_5__Zdata_4273.41__Zactual_4273.57__Zdifference_0.159993__LTSignalUsed_3.txt', delimiter='\t', usecols=(0, 1)).T

# %% merge data into single array

check that the first row matches wavenuber given in first file
IF TRUE
    append return array with second row (y data)
ELSE
    throw mismatch error

return k x n array where:
    k = number of samples
    n = number of y values

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
'''
