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
import numpy as np
import csv

# load wavenumbers for use as x data for each sample
# wavenumbers should be the same for each sample, so data arbitrarily gathered from animal_1_cornea_0
# wavenumbers for each sample are later checked against this array
# generate a 1 x 1 2d array containing labels for x data [wavenumber]
wave_lab = np.array(["wavenumber"])[None, :]  # converts 1d array to 1 x n 2d array
# generate a 1 x n 2d array containing wavenumbers to be used as x data for all samples
wavenumbers = np.genfromtxt(
    Path('../../data/banbury/cornea/cornea_eye_1_fp_sub_0__X_20912.1__Y_-1142.71__Time_0__Zdata_4264.95__'
         'Zactual_4264.89__Zdifference_-0.0527536__LTSignalUsed_3.txt'),
    dtype='float', delimiter='\t', usecols=(0)).T[None, :]


def save_x_data():
    """index x data in column 0 with [wavenumber] and save to data directory and three subdirectories"""
    p = Path('../../data/banbury/_data/by_tissue_by_animal/_wavenumbers.csv')
    q = Path('../../data/banbury/_data/all_tissues_by_animal/_wavenumbers.csv')
    r = Path('../../data/banbury/_data/by_tissue_all_animals/_wavenumbers.csv')
    s = Path('../../data/banbury/_data/_wavenumbers.csv')
    with p.open('w') as f:
        np.savetxt(f, np.hstack((wave_lab, wavenumbers)), delimiter=',', fmt="%s")
    with q.open('w') as f:
        np.savetxt(f, np.hstack((wave_lab, wavenumbers)), delimiter=',', fmt="%s")
    with r.open('w') as f:
        np.savetxt(f, np.hstack((wave_lab, wavenumbers)), delimiter=',', fmt="%s")
    with s.open('w') as f:
        np.savetxt(f, np.hstack((wave_lab, wavenumbers)), delimiter=',', fmt="%s")


class ByTissueByAnimal(object):
    def __init__(self, tissue_type, animal_number):
        """creates separate csv files containing all spectral data obtained from a single tissue from a single animal"""
        self.tissue = tissue_type
        self.animal = animal_number
        self.file_dir = None
        self.data_dir = Path('../../data/banbury/_data/by_tissue_by_animal')
        self.file_dict = {}
        self.x_labels = wave_lab
        self.y_labels = None
        self.x_data = wavenumbers
        self.y_data = None

    def file_search(self):
        """search input tissue subdirectory for all files relating to input animal number
        and return dictionary of {animal_[animal_number]_[tissue]_[iteration_number] : file}"""
        self.file_dir = Path('../../data/banbury/' + self.tissue)
        search_term = f"*eye_{self.animal}_*"  # data exist as eye_[animal_number]_[tissue] and [tissue]_eye_[animal_number]
        for i, file in enumerate(sorted(Path(self.file_dir).glob(search_term))):
            self.file_dict.update({f"animal_{self.animal}_{self.tissue}_{i}": f"{file}"})
        return self.file_dir, self.file_dict

    def generate_y_labels(self):
        """create 1 x n 2d array containing labels for y values [animal_[animal_number]_[tissue]_[iteration_number]]"""
        self.y_labels = np.array(list(self.file_dict.keys()))[:, None]  # converts 1d array to 2d array with 1 row
        return self.y_labels

    def extract_y_data(self):
        """create a k x n 2d array of intensity values
        if the x data of each file k match the global x data stored in wavenumbers
        else an error is thrown"""
        # initialise empty k x n 2d array where k = number of keys in dictionary
        self.y_data = np.empty((len(self.file_dict), self.x_data.shape[1]))
        # for each filepath in file_dict check x data against wavenumbers
        # if True insert y data into row[iteration] of array
        # else throw error that x values do not match
        for i, v in enumerate(self.file_dict.values()):
            xdata = np.genfromtxt(v, dtype='float', delimiter='\t', usecols=0).T
            xdata = xdata[None, :]  # convert 1d array to 2d array with 1 row
            if np.array_equal(wavenumbers, xdata):
                ydata = np.genfromtxt(v, dtype='float', delimiter='\t', usecols=1).T
                ydata = ydata[None, :]  # convert 1d array to 2d array with 1 row
                self.y_data[i, :] = ydata[0, :]  # replace iteration number row of yy with first row of ydata
            else:
                assert "x values do not match"
        return self.y_data

    def save_y_data(self):
        """index y data in column 0 with animal_[animal_number]_[tissue]_[iteration_number]
        and save to by_tissue_by_animal subdirectory"""
        p = Path(self.data_dir / f"animal_{self.animal}_{self.tissue}.csv")
        with p.open('w') as f:
            np.savetxt(f, np.hstack((self.y_labels, self.y_data)), delimiter=',', fmt="%s")

    def do_all_the_things(self):
        self.file_search()
        self.generate_y_labels()
        self.extract_y_data()
        self.save_y_data()


def all_tissues_by_animal(animal_number):
    """creates separate csv files containing all spectral data obtained from all tissues from a single animal"""
    btba_path = Path('../../data/banbury/_data/by_tissue_by_animal/')
    atba_search = f"animal_{animal_number}_*"  # files are named [animal_i_tissue]
    atba_dict = {}
    for i, file in enumerate(sorted(Path(btba_path).glob(atba_search))):
        atba_dict.update({f"animal_{animal_number}_{i}": f"{file}"})
    p = Path('../../data/banbury/_data/all_tissues_by_animal/' + f"animal_{animal_number}.csv")
    with p.open('w') as f:
        writer = csv.writer(f)
        for v in atba_dict.values():
            reader = csv.reader(open(v))
            for row in reader:
                writer.writerow(row)


def by_tissue_all_animals(tissue_type):
    """creates separate csv files containing all spectral data obtained from a single tissue from all animals"""
    btba_path = Path('../../data/banbury/_data/by_tissue_by_animal/')
    btaa_search = f"*_{tissue_type}*"  # files are named [animal_i_tissue]
    btaa_dict = {}
    for i, file in enumerate(sorted(Path(btba_path).glob(btaa_search))):
        btaa_dict.update({f"animal_{i}_{tissue_type}": f"{file}"})
    p = Path('../../data/banbury/_data/by_tissue_all_animals/' + f"{tissue_type}.csv")
    with p.open('w') as f:
        writer = csv.writer(f)
        for v in btaa_dict.values():
            reader = csv.reader(open(v))
            for row in reader:
                writer.writerow(row)


def all_data():
    """creates a single csv file containing all spectral data obtained from all tissues from all animals"""
    btba_path = Path('../../data/banbury/_data/by_tissue_by_animal/')
    ad_search = f"animal*.csv"  # files are named [animal_i_tissue]
    ad_dict = {}
    for i, file in enumerate(sorted(Path(btba_path).glob(ad_search))):
        ad_dict.update({f"sample_{i}": f"{file}"})
    p = Path('../../data/banbury/_data/' "all_data.csv")
    with p.open('w') as f:
        writer = csv.writer(f)
        for v in ad_dict.values():
            reader = csv.reader(open(v))
            for row in reader:
                writer.writerow(row)

"""
# script for running data extraction

# save wavnumbers as a base file in _data directory and each subdirectory

# create dictionary of instance names mapped to possible tissue type and animal number
tissue = ["cornea", "lens", "optic_nerve", "retina", "vitreous"]
run_dict = {}
for i in range(1, 12):
    for t in tissue:
        run_dict.update({f"{t}{i}": [f"{t}", f"{i}"]})

# print list of commands to initialise class instances
for k, v in run_dict.items():
    print(f"{k} = ByTissueByAnimal(\"{v[0]}\", {v[1]})")
# the resultant printed list can be copied into the console to initialise the 55 instances of ByTissueByAnimal

# print list of commands to execute functions of class instances
for key in run_dict.keys():
    print(f"{key}.do_all_the_things()")
# the resultant printed list can be copied into the console to execute the do_all_the_things() function
# of the 55 instances of ByTissueByAnimal

# now all single animal_tissue files are in the by_tissue_by_animal subdirectory

# fill all_tissues_by_animal subdirectory
for x in range(1, 12):
    all_tissues_by_animal(x)

# fill by_tissue_all_animals subdirectory
for tis in tissue:
    by_tissue_all_animals(tis)

# fill all_data file
all_data()
"""