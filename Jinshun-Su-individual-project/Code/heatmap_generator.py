import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, cv2
from PIL import Image
import seaborn as sns
import tqdm

# https://seaborn.pydata.org/generated/seaborn.heatmap.html
# http://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def Heatmap_Generator(CATEGORIES, Path_From, Path_To,label):
    for label_i in label:
        for category in CATEGORIES:
            # print(category)
            data_path = os.path.join(Path_From+label_i, category)
            # print(data_path)
            save_path = os.path.join(Path_To+label_i, category)
            # print(save_path)
            try:
                os.mkdir(save_path)
            except OSError:
                print("Creation of the directory %s failed" % save_path)
            else:
                print("Successfully created the directory %s" % save_path)

            for csv in os.listdir(data_path):
                print('-')
                # print(csv)
                try:
                    # print(csv)
                    matrix_array = pd.read_csv(os.path.join(data_path, csv), header=None)
                    # print(os.path.join(data_path, csv))
                    img_list = matrix_array.values.tolist()
                    # print(np.asarray(img_list).shape)
                    img = sns.heatmap(img_list, vmin=-1.05, vmax=1.05, cmap="gist_rainbow", cbar=False)
                    img.axis('off')
                    img_name = os.path.join(save_path, csv.replace('.csv', '.png'))
                    # print(img_name)
                    plt.savefig(img_name, bbox_inches='tight', pad_inches=0.01)
                    # # plt.show()
                    # print('-')
                except Exception as e:  # in the interest in keeping the output clean...
                    pass


sns.set()
path = 'C:/Users/Administrator/Desktop/M2_Final/Data/'
print(path)
Path_From = path + '/Validation/AE/CNN_22PMU_MissingOneData_add10dB/'
Path_To = path + '/IMAGE/CNN_22PMU_MissingOneData_add10dB/'
CATEGORIES = ['1', '2', '3', '4', '5', '6', '7', '8']
label = ['Train/', 'Test/', 'Val/']
for label_i in label:
    save_path_i = os.path.join(Path_To + label_i)
    try:
        os.mkdir(save_path_i)
    except OSError:
        print("Creation of the directory %s failed" % save_path_i)
    else:
        print("Successfully created the directory %s" % save_path_i)


Heatmap_Generator(CATEGORIES, Path_From, Path_To,label)