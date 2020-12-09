import os
import shutil

def file_name(compare_path,extra_file_path):
    img_list = []
    raw_list = []
    for root, dirs, files in os.walk(compare_path):
        for file in files:
            if os.path.splitext(file)[1] == '.png':   # divided path into file_name and suffix_format
                img_list.append(os.path.splitext(file)[0])
            elif os.path.splitext(file)[1] == '.csv':
                raw_list.append(os.path.splitext(file)[0])

    diff = set(raw_list).difference(set(img_list))  # the group of elements in 'a' but not in 'b'
    print('There are %s images that lack.' % len(diff))
    for name in diff:
        print("no png", name + ".csv")
        original_file = compare_path + name + ".csv"
        move_to_file = extra_file_path + name + ".csv"
        shutil.copyfile(original_file, move_to_file)

    diff2 = set(img_list).difference(set(raw_list))  # the group of elements in 'b' but not in 'a'
    print('There are %s csv files that lack.' % len(diff2))
    for name in diff2:
        print("no csv", name + ".png")

    return img_list, raw_list


if __name__ == '__main__':
    compare_path = '/XXXX/compare/'
    # You need to create a new folder, and copy all .csv and .png files into it, then the path is that folder path

    extra_file_path = '/XXXX/extra_file/'
    file_name(compare_path,extra_file_path)