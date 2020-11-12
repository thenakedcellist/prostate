import pandas as pd
df = pd.read_csv('__PATH__/data_file.csv', header=None)
shuffled_data = df.sample(frac=1)
shuffled_data.to_csv('__PATH__/shuffled_data.csv', header=False, index=False)