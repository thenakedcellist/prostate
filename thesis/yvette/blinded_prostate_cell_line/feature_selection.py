import numpy as np
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('../../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../../data/yvette_20_11_18/shuffled_data_named.csv')
figpath = Path('img_feature_selection/')
datestr = '2021_02_28'

x_data = np.genfromtxt(x_path, delimiter=',')
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1057))

label_list = ['PNT2', 'LNCaP']
marker_list = ['o', 'x']
colour_list = ['#FFA500', '#FF00FF']

# outlier observation
removal_list = [46]

'''
Default values
--------------
x by y: 5*SQRT(n)
sigma:  1.0
learning rate:  0.5
random seed:    1
'''

som = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=1)

som.som_setup(x_data, y_data, y_path, label_list, marker_list, colour_list)
som.remove_observations_from_input_data(removal_list)
som.frobenius_norm_normalisation()
som.train_som(10000)
som.plot_som_scatter(figpath, datestr, onlyshow=True)
som.plot_density_function(figpath, datestr, onlyshow=True)

# feature selection
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
'''
# correlation matrix
data_df = pd.DataFrame(som.ydata, columns=x_data)
data_df.insert(0, "Class", som.t)
feature_names = list(data_df.columns)
feature_df = pd.DataFrame(data_df, columns=data_df.columns)
target_feature = feature_df.iloc[:, 0]
Features = feature_df.iloc[:, 1:]
feat_num = len(feature_names)
corrMatrix = feature_df.corr().round(2)
sn.heatmap(corrMatrix, annot=True)
plt.show()
'''
# component planes
W = som.get_weights()
feature_only_names = list(x_data)
for i, f in enumerate(feature_only_names):
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'{f}')
    ax.pcolor(W[:, :, i].T, cmap='coolwarm')
    ax.set_xticks(np.arange(som.x + 1))
    ax.set_yticks(np.arange(som.y + 1))
    plt.savefig(figpath / f'{f}.png', format='png')

# feature selection method for supervised SOM
'''
def som_feature_selection(W, labels, target_index=0, a=0.04):
    """ Performs feature selection based on a self organised map trained with the desired variables

    INPUTS: W = numpy array, the weights of the map (X*Y*N) where X = map's rows, Y = map's columns, N = number of variables
            labels = list, holds the names of the variables in same order as in W
            target_index = int, the position of the target variable in W and labels
            a = float, an arbitrary parameter in which the selection depends, values between 0.03 and 0.06 work well

    OUTPUTS: selected_labels = list of strings, holds the names of the selected features in order of selection
             target_name = string, the name of the target variable so that user is sure he gave the correct input
    """
    W_2d = np.reshape(W, (W.shape[0] * W.shape[1], W.shape[2]))  # reshapes W into MxN assuming M neurons and N features
    target_name = labels[target_index]

    Rand_feat = np.random.uniform(low=0, high=1, size=(W_2d.shape[0], W_2d.shape[1] - 1))  # create N -1 random features
    W_with_rand = np.concatenate((W_2d, Rand_feat), axis=1)  # add them to the N regular ones
    W_normed = (W_with_rand - W_with_rand.min(0)) / W_with_rand.ptp(0)  # normalize each feature between 0 and 1

    Target_feat = W_normed[:, target_index]  # column of target feature

    # Two conditions to check against a
    Check_matrix1 = abs(np.vstack(Target_feat) - W_normed)
    Check_matrix2 = abs(np.vstack(Target_feat) + W_normed - 1)
    S = np.logical_or(Check_matrix1 <= a, Check_matrix2 <= a).astype(int)  # applies "or" element-wise in two matrices

    S[:, target_index] = 0  # ignore the target feature so that it is not picked

    selected_labels = []
    while True:

        S2 = np.sum(S, axis=0)  # add all rows for each column (feature)

        if not np.any(S2 > 0):  # if all features add to 0 kill
            break

        selected_feature_index = np.argmax(S2)  # feature with the highest sum gets selected first

        if selected_feature_index > (S.shape[1] - (Rand_feat.shape[1] + 1)):  # if random feature is selected kill
            break

        selected_labels.append(labels[selected_feature_index])

        # delete all rows where selected feature evaluates to 1, thus avoid selecting complementary features
        rows_to_delete = np.where(S[:, selected_feature_index] == 1)
        S[rows_to_delete, :] = 0

    #     selected_labels = [label for i, label in enumerate(labels) if i in feature_indices]
    return selected_labels, target_name

selected_features, target_name = som_feature_selection(W, feature_names, 0, 0.04)
print("Target variable: {}\nSelected features {}".format(target_name, selected_features))
'''