import os
import cv2 as cv
from pylab import *
from numpy import *
import pydotplus as pydot

maxsize = (100, 100)
path = 'pub'

# Ham resize lai anh cho cung kich thuoc
def read_file(full_path):

    img_list = []
    i = 0
    for file in os.listdir(full_path):
        path_file = os.path.join(full_path, file)
        img = cv.imread(path_file)

        img_resize = cv.resize(img, maxsize)

        filename = os.path.join(path, str(i) + '.png')
        cv.imwrite(filename, img_resize)  # need temporary files of the right size

        img_list.append(img_resize)
        i = i + 1
    
    return img_list

img_list = read_file(path)
nbr_images = len(img_list)
match_score = zeros((nbr_images, nbr_images))

for i in range(nbr_images):
    for j in range(i, nbr_images): # Chỉ so sánh ở tam giác trên của ma trận, dưới lấy đối xứng
        print("Comparing: ", i, " - ", j)
        sift = cv.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_list[i], None)
        kp2, des2 = sift.detectAndCompute(img_list[j], None)

        # BFMaching 
        bf = cv.BFMatcher(cv.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        # # Excellent maching
        # good_maching = []
        # for n, m in matches:
        #     if n.distance < 0.05 * m.distance:
        #         good_maching.append(n)
        
        nbr_maching = len(matches)

        print('Number of maching: ', nbr_maching)

        match_score[i][j] = nbr_maching

# Copy
for i in range(nbr_images):
    for j in range(i + 1, nbr_images): # Không xét tới đường chéo chính
        match_score[j][i] = match_score[i][j]

# Visualization
threshold = 90  #  At least 2 matching points can be regarded as connection
print(match_score)

g = pydot.Dot(graph_type='graph')  # No digraph is needed

# Pairing
path = 'G:\TuDepTrai\ThucTap\Chapter2\pub'
for i in range(nbr_images):
    for j in range(i + 1, nbr_images):
        if match_score[i, j] > threshold:
            filename = os.path.join(path, str(i) + '.png')
            print(filename)
            g.add_node(pydot.Node(str(i), fontcolor='transparent', shape='rectangle', image=filename))
            filename = os.path.join(path, str(j) + '.png')
            g.add_node(pydot.Node(str(j), fontcolor='transparent', shape='rectangle', image=filename))
            g.add_edge(pydot.Edge(str(i), str(j)))
#Drawing S geographical marker SIFT matching map
g.write_jpg('sift.jpg')

# for file in os.listdir('pub'):
#     if(file.endswith('.png')):
#         file_path = os.path.join('pub', file)
#         os.remove(file_path)

