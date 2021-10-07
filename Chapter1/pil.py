# Đọc ảnh bằng PIL
from PIL import Image
from pylab import *
pil_img = Image.open('cat 2.jpg')

# # # Đọc và convert màu ảnh
# pil_img = Image.open('cat 2.jpg').convert('L') # Convert grayscale color

# imshow(pil_img)
# show()

# # Convert iamge to another format
# import os
# for file in os.listdir('images'):
#     new_file = os.path.splitext(file)[0] + '.png'
#     file = os.path.join('images', file)
#     new_file = os.path.join('images', new_file)
#     if new_file != file:
#         try:
#             Image.open(file).save(new_file)
#         except IOError:
#             print('Can\'t convert file: ', file)

# # Tạo một Thumbnails
# pil_img.thumbnail((128, 128))

# # # Cắt ảnh
# box = (100, 100, 400, 400)
# region = pil_img.crop(box)
# region.show()

# # Copy ảnh
# region = region.transpose(Image.ROTATE_180)
# pil_img.paste(region,box)

# # resize and rotate
out = pil_img.resize((128,128))
out.show()
out = pil_img.rotate(45)
out.show()






