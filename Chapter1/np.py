from numpy import *
from PIL import Image
from pylab import *

im = array(Image.open('cat 2.jpg').convert('L'))
print(im.shape, im.dtype)
# im = array(Image.open('cat 2.jpg').convert('L'),'f')
# print(im.shape, im.dtype)

# i = 2, j = 4
# im[i,:] = im[j,:] # set the values of row i with values from row j
# im[:,i] = 100 # set all values in column i to 100
# im[:100,:50].sum() # the sum of the values of the first 100 rows and 50 columns
# im[50:100,50:100] # rows 50-100, columns 50-100 (100th not included)
# im[i].mean() # average of row i
# im[:,-1] # last column
# im[-2,:] # second to last row
# im[-2] # second to last row

# 2. Phép chuyển đổi ở mức xám
im2 = 255 - im # chuyển ảnh xám, từ tối thành sáng, sáng thành tối
im3 = (100.0/255) * im + 100 # Ép giá trị pixel trong khoảng từ 100...200
im4 = 255.0 * (im/255.0)**2 # Hàm bình phương
print(im2)

fig, axs = subplots(1, 4, figsize=(10,8))
gray()
axis('equal')
axis('off')
axs[0].set_title('Origin Image')
axs[0].imshow(im)
axs[1].set_title('255 - im')
axs[1].imshow(im2)
axs[2].set_title('Pixel in [100, 200]')
axs[2].imshow(im3)
axs[3].set_title('Square')
axs[3].imshow(im4)
show()


# Chuyển từ array qua image
# pil_im = Image.fromarray(im)
