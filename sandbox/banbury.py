"""

----------------------------------------------------

Directory notes
This directory contains data from the banbury directories
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
open all files for unique animal_tissue
    set row 0 of sample 0 as row 0 of output_data
    compare row 0 of file k with row 0 of output_data
        IF same
            appended row 1 of file k to output_data
        ELSE
            throw error "wavenumbers do not match"
save csv file in directory matching file name animal_tissue
Combine animal_tissue files into single animal and tissue files and place in relevant directories
Combine animal files into single data file and place im root

review banbury source code to see how they got their average spectra
define functions to plot spectra and average spectra
    method for single spectrum
    method for average spectrum

"""

from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt


class ByTissueByAnimal(object):
    def __init__(self, tissue_type, animal_number):
        """creates separate csv files containing all spectral data obtained from a single tissue from a single animal"""
        self.tissue = tissue_type
        self.animal = animal_number
        self.file_dir = None
        self.file_dict = {}
        self.x_data = None
        self.y_data = None

    def file_search(self):
        """search tissue directory for all files relating to input tissue and animal number
        and return dictionary of {animal_[animal_number]_[tissue]_[iteration_number] : file}"""
        self.file_dir = Path('../data/banbury/' + self.tissue)
        search_term = '*eye_{}_*'.format(self.animal)  # data exist as eye_[animal_number]_[tissue] and [tissue]_eye_[animal_number]_
        i = 1
        for file in Path(self.file_dir).glob(search_term):
            self.file_dict.update({f"animal_{self.animal}_{self.tissue}_{i}": f"{file}"})
            i = i + 1
        return self.file_dir, self.file_dict

    def x_data_extraction(self):
        """search file corresponding to first key in dictionary and extract first column,
        return this data as the first row of a np array"""
        self.x_data = np.genfromtxt(self.file_dict[f"animal_{self.animal}_{self.tissue}_1"], dtype='float', delimiter='\t', usecols=(0)).T  # generates first row of data from first column of the file pointed to in the first dictionary key
        return self.x_data

"""check input banbury data are in expected column[0] wavenumber column[1] value format,
        transpose to row[0] wavenumber row[1] value format,
        check that row[0] is same for each file
        create ouput dictionary with row[0] wavenumber row[1-k] value format
        and labels animal_[animal_number]_[tissue]_[iteration_number]"""


'''
test script
cornea1 = ByTissueByAnimal("cornea", 1)
cornea1.file_search()
cornea1.x_data_extraction()
'''



#    h = input_x.keys()
 #   x = np.genfromtxt(key, delimiter='\t').T
  #  for key in h:
   #     x = np.genfromtxt(key, delimiter='\t').T
   #     y = np.array()
    #    if x == x_data:



'''
x data to be first row of first item.T (data needs to be transposed) with wavenumber in first cell
then for every entry in dictionary, take file.T, check row 0 = x data
then append row 1 to y data
assign key to column 1 of row
can list keys in a dictionary to call every element
'''











class AllTissuesByAnimal:
    """creates separate csv files containing all spectral data obtained from all tissues from a single animal"""
    pass


class ByTissueAllAnimals:
    """creates separate csv files containing all spectral data obtained from a single tissue from all animals"""
    pass


class AllData:
    """creates a single csv file containing all spectral data obtained from all tissues from all animals"""
    pass

