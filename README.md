Autoencoder-as-a-Classifier-using-IEEE-34-Test-Feeder-PMU-Dataset

The links for the dataset we used in this project:
1. Dataset for ideal model:
https://drive.google.com/file/d/1kpE5WM8xH4sZ2UHLmPOBQYgIfQqPZXM9/view?usp=sharing

   => It has to be unzipped under Code/PMU_PU

2. Datasets for non-ideal models:
https://drive.google.com/file/d/1nhcpW6uGyAgIubvCfTApjpxKym0YJj-U/view?usp=sharing

   => It has to be unzipped under Code/Validation/AE_Train&Test

3. Interfered datasets for testing all models:
https://drive.google.com/file/d/1jCASPJ8UeOcz_9YnyDhxv14OkIpZFLFT/view?usp=sharing

   => It has to be unzipped under Code/Validation/Test_Model

First, create a folder for this project.

Then download the three groups of data via links above.

Unzip the .zip files on the instruction folders above.

Open the "Code" folder and find the "train_idealmodel_group4.py" for training the ideal model.

The rest two "train_nonidealmodel_22PMU_10dB_group4.py" and "train_nonidealmodel_22PMU_MissingOneData_10dB_group4.py" are for training two non-ideal models.

The "predict_group4.py" is for testing different groups of data on the specific model and check the prediction accuracy.

PS: If you use the dataset from this project for research, please cite
-Y. Li. <em>Date-Driven Topology Identification in Power Distribution Systems with Machine Learning</em>. Diss. The George Washington University, 2020.
