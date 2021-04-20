import numpy as np
import pandas as pd

# parameters dataset with total error and error difference
df = pd.read_csv('known_soms_parameters.csv', header=0)
df['total_err'] = df['q_err'] + df['t_err']
df['err_diff'] = abs(df['q_err'] - df['t_err'])

# median quantisation errors
q_err_split_array = np.array(np.split(df["q_err"], 8))
quant_errors = pd.DataFrame([[(i + 1), np.min(j), np.max(j), np.median(j)] for i, j in enumerate(q_err_split_array)],
                            columns=['family', 'min_quant_err', 'max_quant_err', 'med_quant_err'])
q_err_med = quant_errors.nsmallest(1, 'med_quant_err', keep='all')
q_err_med_list = sorted(q_err_med['family'].values.tolist())

# median topographic errors
t_err_split_array = np.array(np.split(df["t_err"], 8))
top_errors = pd.DataFrame([[(i + 1), np.min(j), np.max(j), np.median(j)] for i, j in enumerate(t_err_split_array)],
                            columns=['family', 'min_top_err', 'max_top_err', 'med_top_err'])
t_err_med = top_errors.nsmallest(1, 'med_top_err', keep='all')
t_err_med_list = sorted(t_err_med['family'].values.tolist())

# median total errors
tot_err_split_array = np.array(np.split(df["total_err"], 8))
total_errors = pd.DataFrame([[(i + 1), np.min(j), np.max(j), np.median(j)] for i, j in enumerate(tot_err_split_array)],
                            columns=['family', 'min_total_err', 'max_total_err', 'med_total_err'])
tot_err_med = total_errors.nsmallest(1, 'med_total_err', keep='all')
tot_err_med_list = sorted(tot_err_med['family'].values.tolist())

# median error difference
err_diff_split_array = np.array(np.split(df["err_diff"], 8))
error_differences = pd.DataFrame([[(i + 1), np.min(j), np.max(j), np.median(j)] for i, j in enumerate(err_diff_split_array)],
                            columns=['family', 'min_err_diff', 'max_err_diff', 'med_err_diff'])
err_diff_med = error_differences.nsmallest(1, 'med_err_diff', keep='all')
err_diff_med_list = sorted(err_diff_med['family'].values.tolist())

# print indices lists
print("quantisation errors " + str(q_err_med_list))
print("topographic errors " + str(t_err_med_list))
print("total errors " + str(tot_err_med_list))
print("error difference " + str(err_diff_med_list))

# return selected indices from df
ordered_param_list = sorted(list(set(tot_err_med_list) | set(err_diff_med_list)))
search_list = [f'{q}A' for q in ordered_param_list]
selected_data = df[df['family_member'].isin(search_list)]
