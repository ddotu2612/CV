from PIL import Image
from pylab import *

# read image to array
im = array(Image.open('images/cat 2.jpg').convert('L'))

# 1. show image, plot image array, point, line
# plot the image
imshow(im)

# some points
x = [100,100,400,400]
y = [200,500,200,500]

# plot the points with red star-markers
plot(x, y, 'r*')

# line plot connecting the first two points
plot(x[:2],y[:2])

# add title and show the plot
title('Plotting: "empire.jpg"')
axis('off') # Off axis
show()

plot(x,y) # default blue solid line
plot(x,y,'r*') # red star-markers
plot(x,y,'go-') # green line with circle-markers
plot(x,y,'ks:') # black dotted line with square-markers

# 2. Image contours and histograms
# create a new figure
figure()
# don’t use colors
gray()
# show contours with origin upper left corner
contour(im, origin='image')
axis('equal')
axis('off')

figure()
hist(im.flatten(),128)
show()

# 3. Chú thích tương tác
imshow(im)
print('Please click 3 points')
x = ginput(3) # cho phép chọn 3 điểm trên ảnh
print('you clicked:', x) # in toạ độ từng điểm ra màn hình
show()


