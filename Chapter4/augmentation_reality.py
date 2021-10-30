import objloader
from OpenGL.GL import *
from OpenGL.GLU import *
from numpy.lib.function_base import i0
import pygame
import pygame.image
from pygame.locals import *
import numpy as np
import math
from OpenGL.GLUT import *
import pickle

width, height = 1000, 747


def set_projection_from_camera(K):
    """ Thiet lap view tu mo ma tran hieu chinh camera. """

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()  # Khoi tao lai nhung thay doi phia truoc

    fx = K[0][0]
    fy = K[1][1]
    fovy = 2*np.arctan(0.5 * height / fy)*180 / math.pi
    aspect = (width*fy)/(height*fx)

    # Định nghĩa mặt phẳng cắt gần và xa
    near = 0.1
    far = 100.0

    # set perspective
    gluPerspective(fovy, aspect, near, far)
    glViewport(0, 0, width, height)


def set_modelView_from_camera(Rt):
    """ Tạo ma trận model view từ camera pose. """

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Xoay ấm trà 90 độ quanh trục x để trục z hướng lên
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # Đặt độ xoay tới giá trị xấp xỉ tốt nhất
    R = Rt[:, :3]
    U, S, V = np.linalg.svd(R)
    R = np.dot(U, V)
    R[0, :] = -R[0, :]  # thay đổi dấu của trục x

    # set translation
    t = Rt[:, 3]

    # setup 4*4 model view matrix
    M = np.eye(4)
    M[:3, :3] = np.dot(R, Rx)
    M[:3, 3] = t

    # transpose and flatten to get column order
    M = M.T
    m = M.flatten()

    # replace model view with the new matrix
    glLoadMatrixf(m)


def draw_background(imname):
    """ Vẽ ảnh background sử dụng một quad. """

    # Load background image (.bmp) to OpenGL texture
    bg_image = pygame.image.load(imname).convert()
    bg_data = pygame.image.tostring(bg_image, 'RGBX', 1)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # bind the texture
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, glGenTextures(1))
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_data)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    # create quad to fill the whole window
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(1.0, 1.0, -1.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glEnd()

    # clear the texture
    glDeleteTextures(1)


def draw_teapot(size):
    """ Draw a red teapot at the origin. """
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)

    # draw red teapot
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0, 0, 0, 0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.0, 0.0, 0.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.7, 0.6, 0.6, 0.0])
    glMaterialfv(GL_FRONT, GL_SHININESS, 0.25*128.0)
    glutSolidTeapot(size)


def setup():
    """ Setup window and pygame environment. """
    pygame.init()
    pygame.display.set_mode((width, height), OPENGL | DOUBLEBUF)
    pygame.display.set_caption('OpenGL AR demo')

def load_and_draw_model(filename):
    """ Loads a model from an .obj file using objloader.py.
    Assumes there is a .mtl material file with the same name. """

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)

    # set model color
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0, 0, 0, 0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.75, 1.0, 0.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 0.25*128.0)

    # load from a file
    import objloader
    obj = objloader.OBJ(filename, swapyz=True)
    glCallList(obj.gl_list)

# load camera data
with open('ar_camera.pkl', 'rb') as f:
    K = pickle.load(f)
    Rt = pickle.load(f)

setup()
draw_background(r'G:\DepTrai\ThucTap\img\data\book_perspective.bmp')
set_projection_from_camera(K)
set_modelView_from_camera(Rt)
# draw_teapot(0.02)
load_and_draw_model(r'G:\DepTrai\ThucTap\Chapter4\housemodel\extincteur_obj.obj')

while True:
    event = pygame.event.poll()
    if event.type in (QUIT, KEYDOWN):
        break
    pygame.display.flip()

