import pandas as pd

df1 = pd.read_csv('../../../../data/banbury/_data_1-6/all_data.csv', header=None)
df2 = pd.read_csv('../../../../data/banbury/_data_7-11/all_data.csv', header=None)

merged_data = pd.concat((df1, df2), axis=0)

merged_data.to_csv('../../../../data/banbury/_all_animals/all_animal_data.csv', header=False, index=False)