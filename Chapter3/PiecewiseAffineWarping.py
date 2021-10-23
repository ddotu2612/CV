import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data


image = data.astronaut()
rows, cols = image.shape[0], image.shape[1]

src_cols = np.linspace(0, cols, 20)
src_rows = np.linspace(0, rows, 10)
src_rows, src_cols = np.meshgrid(src_rows, src_cols) # Trả về hai ma trận, src_rows có 10 dòng như nhau, src_column có 20 cột như nhau
src = np.dstack([src_cols.flat, src_rows.flat])[0] # Ghép theo chiều dọc, tạo thành một mảng [[[]]], nên lấy phần tử đầu tiên

# Thêm giao động sin vào toạ độ hàng
dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
dst_cols = src[:, 0]
dst_rows *= 1.5
dst_rows -= 1.5 * 50
dst = np.vstack([dst_cols, dst_rows]).T

tform = PiecewiseAffineTransform()
tform.estimate(src, dst)

out_rows = image.shape[0] - 1.5 * 50
out_cols = cols
out = warp(image, tform, output_shape=(out_rows, out_cols))
print(out_rows, out_cols)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(out)
ax[0].plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
ax[0].axis((0, out_cols, out_rows, 0))
ax[1].imshow(image)
plt.show()