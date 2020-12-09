# Sorting the data into three folders
# Copy the data into a new folder first
import os, random, shutil


def moveFile_Train(fileDir):
        pathDir = os.listdir(fileDir)    # Read the original path of the file
        filenumber = len(pathDir)
        rate = 0.8     # Proportion of extracted files
        picknumber = int(filenumber*rate)   # Extract files in proportion
        sample = random.sample(pathDir, picknumber)   # Randomly select a proportional number of files
        print(sample)
        for name in sample:
                shutil.move(fileDir+'/'+name, tarDir+'/'+name)    # shutil.copyfile(fileDir+name, tarDir+name)
        return

def moveFile_Test(fileDir):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)
    rate = 0.5       # Proportion of the left files
    picknumber = int(filenumber * rate)
    sample = random.sample(pathDir, picknumber)
    print(sample)
    for name in sample:
        shutil.move(fileDir+'/'+name, tarDir+'/'+name)
    return

def moveFile_Val(fileDir):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)
    rate = 1
    picknumber = int(filenumber * rate)
    sample = random.sample(pathDir, picknumber)
    print(sample)
    for name in sample:
        shutil.move(fileDir+'/'+name, tarDir+'/'+name)
    return


if __name__ == '__main__':
    for num in range(1, 8):
        path_Train = '/XXXX/train/' + str(num)
        path_Test = '/XXXX/test/'+str(num)
        path_Val = '/XXXX/val/' + str(num)
        try:
            os.mkdir(path_Train)   # os.makedirs(path_Train)
            os.mkdir(path_Test)
            os.mkdir(path_Val)
        except OSError:
            print("Creation of the directory %s failed" % path_Train)
            print("Creation of the directory %s failed" % path_Test)
            print("Creation of the directory %s failed" % path_Val)
        else:
            print("Successfully created the directory %s" % path_Train)
            print("Successfully created the directory %s" % path_Test)
            print("Successfully created the directory %s" % path_Val)

    for num in range(1, 8):
        fileDir = "/XXXX/train/"+str(num)    # Source file folder path
        tarDir = '/XXXX/'+str(num)             # Move to new folder path
        moveFile_Train(fileDir)       # move to Train folder

    for num in range(1, 8):
        fileDir = "/XXXX/test/"+str(num)
        tarDir = '/XXXX/'+str(num)
        moveFile_Test(fileDir)       # move to Test folder

    for num in range(1, 8):
        fileDir = "/XXXX/val/"+str(num)
        tarDir = '/XXXX/'+str(num)
        moveFile_Val(fileDir)       # move to Validation folder