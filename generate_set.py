from numpy import random
import glob
import os

def train_set_random_generate(dataset_path,destination_dir):
    file_train = open(os.path.join(destination_dir,'train.csv'), 'w')
    file_test =  open(os.path.join(destination_dir,'val.csv'), 'w')
    file_train.write('fname,labels\n')
    file_test.write('fname,labels\n')
    # Populate train.txt and test.txt
    counter = 1
    index_test = 10
    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.wav")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        if counter == index_test+1:
            counter = 1
            file_test.write(title+'.wav,')
            file_test.write(str(random.randint(0,2)))
            file_test.write('\n')
        else:
            file_train.write(title+'.wav,')
            file_train.write(str(random.randint(0,2)))
            file_train.write('\n')
            counter = counter + 1


datadir = 'D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05'
destdir = 'D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05'
train_set_random_generate(datadir,destdir)
