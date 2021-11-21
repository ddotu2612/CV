from numpy.core.fromnumeric import nonzero
from scipy.linalg import expm, rq, det, inv
from numpy import *
from numpy import sign

class Camera(object):
    """ Class đại diện cho pin-hole cameras. """

    def __init__(self, P):
        """ Khởi tạo P=L=K[R|t] camera model. """
        self.P = P # camera metrix (project metrix) là ma trận 3 * 4
        self.K = None # Calibration metrix
        self.t = None # translation
        self.c = None # center camera
        self.R = None # rotation
    
    def project(self, X):
        """ Chiếu các điểm trong X (4*n) và chuẩn hoá toạ độ """

        x = dot(self.P, X)
        for i in range(3):
            x[i] /= x[2]
        return x

    # Hàm tính factor
    def factor(self):
        """ Tách K, R, t từ P = K[R|t]. """
        
        # factor firsr 3*3 part
        K, R = rq(self.P[:,:3])

        # make diagonal of K positive
        T = diag(sign(diag(K)))
        if det(T) < 0:
            T[1, 1] *= -1
        
        self.K = dot(K, T)
        self.R = dot(K, R)
        self.t = dot(inv(self.K), self.P[:, 3])

        return self.K, self.R, self.t
    
    def center(self):
        if self.c is not None:
            return self.c
        else:
            self.factor()
            self.c = -dot(self.R.T, self.t)
            return self.c

def rotation_matrix(a):
    """ Tạo một ma trận xoay 3D cho việc xoay
    xunh quanh một trục của vector a. """
    R = eye(4)
    R[:3, :3] = expm([[0, -a[2], a[1]], 
    [a[2], 0, -a[0]], [-a[1], a[0], 0]])

    return R

def my_calibration(sz):
    """ Nhận sz là kích thước của hình ảnh  khác. """
    row, col = sz
    # Chia tỉ lệ nếu hình ảnh nhỏ hơn
    fx = 2555*col/2592
    fy = 2586*row/1936

    K = diag([fx, fy, 1])
    K[0, 2] = 0.5*col
    K[1, 2] = 0.5*row
    return K