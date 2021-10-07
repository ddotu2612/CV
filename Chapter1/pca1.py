from PIL import Image
from numpy import *
from pylab import *
from imtools import get_imlist, pca

imlist = get_imlist('images')
print(imlist)
im = array(Image.open(imlist[0]).convert('L')) # open one image to get size
m, n = im.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(im).convert('L')).flatten() for im in imlist], 'f')
# perform PCA
V, S, immean = pca(immatrix)
# show some images (mean and 7 first modes)
figure()
gray()
subplot(2, 4, 1)
imshow(immean.reshape(m, n)) # Chuyển lại hình ảnh từ dạng flatten
for i in range(7):
    subplot(2, 4, i+2)
    imshow(V[i].reshape(m, n))
show()

# Sử dụng Pickle để lưu lại các đối tượng cho lần dùng sau
import pickle 
f = open('font_pca_modes.pkl', 'wb')
pickle.dump(immean,f)
pickle.dump(V,f)
f.close()

# Load 
f = open('font_pca_modes.pkl', 'wb')
immean = pickle.load(f)
V = pickle.load(f)
f.close()

# Thao tác với file không cần đóng file
# open file and save
with open('font_pca_modes.pkl', 'wb') as f:
    pickle.dump(immean,f)
    pickle.dump(V,f)
# open file and load
with open('font_pca_modes.pkl', 'wb') as f:
    immean = pickle.load(f)
    V = pickle.load(f)

# Lưu một mảng X trong file txt
from numpy import savetxt, loadtxt
x = [1, 2, 3, 4]
savetxt('test.txt', x, '%i')
# Load file txt
x = loadtxt('test.txt')