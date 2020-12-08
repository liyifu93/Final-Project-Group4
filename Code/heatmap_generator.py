import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, cv2
from PIL import Image
import seaborn as sns

# https://seaborn.pydata.org/generated/seaborn.heatmap.html
# http://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def Heatmap_Generator(CATEGORIES, Path_From, Path_To, Folders):
    for f in Folders:
        for category in CATEGORIES:
            # print(category)
            data_path = os.path.join(Path_From + f, category)
            # print(data_path)
            save_path = os.path.join(Path_To + f, category)
            # print(save_path)

            try:
                os.mkdir(save_path)
            except OSError:
                print("Creation of the directory %s failed" % save_path)
            else:
                print("Successfully created the directory %s" % save_path)

            i = 0
            for csv in os.listdir(data_path):
                print(i)
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

                i = i + 1

sns.set()
path = '/home/ubuntu/DATS6203/Final_Project/'
print(path)
Path_From = path + 'Validation/AE_Train&Test/CNN_22PMU_10dB/'
Path_To = path + 'Validation/AE_Train&Test/CNN_22PMU_10dB_HeatMap/'
CATEGORIES = ['7', '8']
# folders = ['Train/', 'Test/', 'Val/']
folders = ['Train/']
Heatmap_Generator(CATEGORIES, Path_From, Path_To, folders)