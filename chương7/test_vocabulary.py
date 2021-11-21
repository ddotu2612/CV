import pickle
import vocabulary
import os
from PIL import Image
import numpy as np

def create_images(path):
    imlist = []

    for i, file in enumerate(os.listdir(path)):
        print(f'Dang xu ly file thu {i + 1} : {file}')
        fullfile = os.path.join(path, file)
        im = np.array(Image.open(fullfile).convert('L'))
        imlist.append(im)
    
    with open('imlist.pkl', 'wb') as f:
        pickle.dump(imlist, f)

def create_imname(path):
    imname = []

    for i, file in enumerate(os.listdir(path)):
        print(f'Dang xu ly file thu {i + 1} : {file}')
        fullfile = os.path.join(path, file)
        imname.append(fullfile)
    
    with open('imname.pkl', 'wb') as f:
        pickle.dump(imname, f)

path = r'G:\dvtu\ThucTap\img\ukbench'
# create_images(path)
create_imname(path)

# voc = vocabulary.Vocabulary('ukbenchtest')
# voc.train('imlist.pkl', 1000, 10)

# # saving vocabulary
# with open('vocabulary.pkl', 'wb') as f:
#     pickle.dump(voc, f)
# print('vocabulary is:', voc.name, voc.nbr_words)