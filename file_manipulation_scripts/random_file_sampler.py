# %% setup

import os
import csv
import random
import shutil


# %% define random file creator and random file sampler

# function to create sequential dummy files
def file_create(files_list, source):
    os.chdir(source)  # set directory
    for item in files_list:
        with open("{}.csv".format(item), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Wavnumber", "Measurement"])
            writer.writerow([1, 73, random.randint(1, 10000)])  # inclusive range

# function to copy x random files into new directory per iteration
def file_sample(sample_number, iterations):
    dir_list = list(range(1, iterations+1))  # exclusive range
    dir_list=list( map(str, dir_list))
    for i in dir_list:
        os.mkdir(os.path.join(root_path, 'test_target_' + i))  # make new directory for iteration
        dest = os.path.join(root_path, 'test_target_' + i)
        # select x random files from source A
        random_file=random.sample(os.listdir(source_a), sample_number)
        for k in random_file:
            file="%s\%s"%(source_a, k)
            shutil.copy(file, dest)
        # sleect x random files from source B
        random_file=random.sample(os.listdir(source_b), sample_number)
        for k in random_file:
            file="%s\%s"%(source_b, k)
            shutil.copy(file, dest)


# %% define programme parameters

# define seqeunce for dummy files
files_list_a = list(range(1, 31))  # exclusive range
files_list_b = list(range(31, 61))  # exclusive range

# define source directories
source_a = r'H:\MSc\project\random_file_sampler\test_A'
source_b = r'H:\MSc\project\random_file_sampler\test_B'

# define root path
root_path = r'H:\MSc\project\random_file_sampler'


# %% run programme

# create dummy files in specified directories
file_create(files_list_a, source_a)
file_create(files_list_b, source_b)

# randomly sample x files from each source folder for y iterations
file_sample(5, 2)
