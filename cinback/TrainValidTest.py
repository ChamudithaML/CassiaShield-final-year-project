import os
import shutil
import random

# os.chdir('Data/train')

def file_copy_test():
    folder_list = os.listdir('Dataz')
    no_of_files=70

    for folder_name in folder_list:
        source = 'Dataz/'+folder_name
        dest='Data/test/'+folder_name

        files=os.listdir(source)

        for file_name in random.sample(files,no_of_files):
            shutil.move(os.path.join(source,file_name),dest)

def file_copy_valid():
    folder_list = os.listdir('Dataz')
    no_of_files=140

    for folder_name in folder_list:
        source = 'Dataz/'+folder_name
        dest='Data/valid/'+folder_name

        files=os.listdir(source)

        for file_name in random.sample(files,no_of_files):
            shutil.move(os.path.join(source,file_name),dest)

def file_copy_train():
    folder_list = os.listdir('Dataz')
    no_of_files=490

    for folder_name in folder_list:
        source = 'Dataz/'+folder_name
        dest='Data/train/'+folder_name

        files=os.listdir(source)

        for file_name in random.sample(files,no_of_files):
            shutil.move(os.path.join(source,file_name),dest)

def folder_make():
    files = os.listdir('Dataz')
    print(files)
    
    for i in files:
        if os.path.isdir(f'Data/train/{i}') is False:
            os.makedirs(f'Data/train/{i}')
        if os.path.isdir(f'Data/test/{i}') is False:
            os.makedirs(f'Data/test/{i}')
        if os.path.isdir(f'Data/valid/{i}') is False:
            os.makedirs(f'Data/valid/{i}')




# folder_make()
# file_copy_train()
# file_copy_valid()
# file_copy_test()








