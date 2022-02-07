from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

class Camera(object):
    """ Class for representing pin-hole cameras. """
    def __init__(self,P):
        """ Initialize P = K[R|t] camera model. """
        self.P = P
        self.K = None # calibration matrix
        self.R = None # rotation
        self.t = None # translation
        self.c = None # camera center
    def project(self,X):
        """ Project points in X (4*n array) and normalize coordinates. """
        x = np.dot(self.P,X)
        for i in range(3):
            x[i] /= x[2]
        return x
    
    def factor(self):
        """ Factorize the camera matrix into K,R,t as P = K[R|t] """

        # factor first 3*3 part
        K, R = linalg.rq(self.P[:,:3])

        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1,1] *= -1

        self.K = np.dot(K,T)
        self.R = np.dot(T,R) # T is its own inverse
        self.t = np.dot(linalg.inv(self.K), self.P[:, 3])

        return self.K, self.R, self.t

    def center(self):
        """ Compute and return the camera center """

        if self.c is not None:
            return self.c

        else:
                       # compute c by factoring
            self.factor()
            self.c = -np.dot(self.R.T, self.t)
            return self.c

def rotation_matrix(a):
    """ Creates a 3D rotation matrix for rotation around the axis of the vector a """

    R = np.eye(4)
    R[:3, :3] = linalg.expm([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return R

points = np.loadtxt('./Examples/Model_House/house.p3d')

points = points.T
points = np.vstack((points, np.ones(points.shape[1])))

# setup camera
P = np.hstack((np.eye(3), np.array([[0],[0],[-10]])))

cam = Camera(P)

# project the points
x = cam.project(points)

#plot projection
plt.figure()
plt.plot(x[0], x[1], 'k.')
plt.show()

# create transformation
r = 0.05 * np.random.rand(3)
rot = rotation_matrix(r)

# rotate camera and project
plt.figure()
for t in range(20):
    cam.P = np.dot(cam.P, rot)
    x = cam.project(points)
    plt.plot(x[0], x[1], 'k.')

plt.show()

def my_calibration(sz):
    row,col = sz
    fx = 2555*col/2592 #2592 is the camera resolution in width
    fy = 2586*row/1936 #1936 is the camera resolution in height
    K = np.diag([fx,fy,1])
    K[0,2] = 0.5*col
    K[1,2] = 0.5*row
    print(K)
    return K

from PCV.geometry import homography
from PCV.localdescriptors import sift
import cv2

# because the sift from PCV.localdescriptors didnt work i had to use SIFT from cv2, this is why the upcoming code differs a bit form the book
# instead of sift.read_feature_from_file cv2.detectAndCompute was used
sift_CV2 = cv2.xfeatures2d.SIFT_create()

# load image
img1 = cv2.imread('./Test-Images/book_frontal.JPG')
img2 = cv2.imread('./Test-Images/book_perspective.JPG')

#img1 = cv2.imread('./Test-Images/IMG_01.JPG')
#img2 = cv2.imread('./Test-Images/IMG_02.JPG')

# find keypoints and descriptors with SIFT
kp1, des1 = sift_CV2.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
kp2, des2 = sift_CV2.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

# find point matches
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Apply Lowe's SIFT matching ratio test
good = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good.append(m)

src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])

# Constrain matches to fit homography
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
#mask = mask.ravel()


def cube_points(c,wid):
    """ Creates a list of points for plotting
    a cube with plot. (the first 5 points are
    the bottom square, some sides repeated). """
    p = []
    #bottom
    p.append([c[0]-wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]-wid,c[2]-wid]) #same as first to close plot
    #top
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]-wid,c[2]+wid]) #same as first to close plot
    #vertical sides
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])
    return np.array(p).T


# camera calibration
K = my_calibration((747,1000)) 
# 3D points at plane z=0 with sides of length 0.2
box = cube_points([0,0,0.1],0.1)
# project bottom square in first image
cam1 = Camera( np.hstack((K,np.dot(K,np.array([[0],[0],[-1]])) )) )
# first points are the bottom square
box_cam1 = cam1.project(homography.make_homog(box[:,:5]))
# use H to transfer points to the second image
box_trans = homography.normalize(np.dot(H,box_cam1))
# compute second camera matrix from cam1 and H
cam2 = Camera(np.dot(H,cam1.P))
A = np.dot(linalg.inv(K),cam2.P[:,:3])
A = np.array([A[:,0],A[:,1],np.cross(A[:,0],A[:,1])]).T
cam2.P[:,:3] = np.dot(K,A)
# project with the second camera
box_cam2 = cam2.project(homography.make_homog(box))
# test: projecting point on z=0 should give the same
point = np.array([1,1,0,1]).T


from PIL import Image	


im0 = np.array(Image.open('./Test-Images/book_frontal.JPG'))
im1 = np.array(Image.open('./Test-Images/book_perspective.JPG'))
# 2D projection of bottom square
plt.figure()
plt.imshow(im0)
plt.plot(box_cam1[0,:],box_cam1[1,:],linewidth=3)
# 2D projection transferred with H
plt.figure()
plt.imshow(im1)
plt.plot(box_trans[0,:],box_trans[1,:],linewidth=3)
# 3D cube
plt.figure()
plt.imshow(im1)
plt.plot(box_cam2[0,:],box_cam2[1,:],linewidth=3)
plt.show()

import pickle

with open('ar_camera.pkl','w') as f:
    pickle.dump(K,f)
    pickle.dump(np.dot(linalg.inv(K),cam2.P),f)


from OpenGL.GL import *             # includes most functions we need
from OpenGL.GLU import *            # the OpenGL Utility library and contains some higher-level functionality (here: used to setup the camera projection)
import pygame, pygame.image         # 1. sets up the window and event controls / 2. e is used for loading image and creating OpenGL textures
from pygame.locals import *  



import math

height = 747
width = 1000

def set_projection_from_camera(K):
    
    """ Set view from a camera calibration matrix. """
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    fx = K[0,0]
    fy = K[1,1]
    
    fovy = 2*np.arctan(0.5*height/fy)*180/math.pi
    aspect = (width*fy)/(height*fx)

    # define the near and far clipping planes
    near = 0.1
    far = 100.0

    # set perspective
    gluPerspective(fovy,aspect,near,far)
    glViewport(0,0,width,height)

def set_modelview_from_camera(Rt):
    """ Set the model view matrix from camera pose. """
    #switch to work on the GL_MODELVIEW matrix and reset it
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # rotate teapot 90 deg around x-axis so that z-axis is up
    Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])

    # set rotation to best approximation
    R = Rt[:,:3]
    U,S,V = linalg.svd(R)   #make sure that its been a ratation matrix
    R = np.dot(U,V)         #the best rotation matrix approximation isgiven by R = UV^T
    R[0,:] = -R[0,:] # change sign of x-axis

    # set translation / flip the x-axis around 
    t = Rt[:,3]
    
    # setup 4*4 model view matrix
    M = np.eye(4)
    M[:3,:3] = np.dot(R,Rx)
    M[:3,3] = t
    
    # transpose and flatten to get column order
    M = M.T
    m = M.flatten()
    
    # replace model view with the new matrix
    glLoadMatrixf(m)

def draw_background(imname):
    """ Draw background image using a quad. """
    # load background image (should be .bmp) to OpenGL texture
    bg_image = pygame.image.load(imname).convert()
    bg_data = pygame.image.tostring(bg_image,"RGBX",1)

    # reset model and clear the color and depth buffer
    glMatrixMode(GL_MODELVIEW)                      
    glLoadIdentity()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    #bind the texture
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D,glGenTextures(1))
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,bg_data)
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)

    # create quad to fill the whole window (corners at -1 and 1 in both dimensions)
    # Note that the coordinates in the texture image goes from 0 to 1
    glBegin(GL_QUADS)
    glTexCoord2f(0.0,0.0); glVertex3f(-1.0,-1.0,-1.0)
    glTexCoord2f(1.0,0.0); glVertex3f( 1.0,-1.0,-1.0)
    glTexCoord2f(1.0,1.0); glVertex3f( 1.0, 1.0,-1.0)
    glTexCoord2f(0.0,1.0); glVertex3f(-1.0, 1.0,-1.0)
    glEnd()

    # clear the texture, so it doesn't interfere with what we want to draw later
    glDeleteTextures(1)



from OpenGL.GLUT import *


# The following function will set up the color and properties to make a pretty red teapot
def draw_teapot(size):
    """ Draw a red teapot at the origin. """
  
    # glEnable is used to turn on OpenGL features
    glEnable(GL_LIGHTING) # enable lightning
    glEnable(GL_LIGHT0) # enable light
    glEnable(GL_DEPTH_TEST) # depth testing is turned on so that objects are rendered according to their depth
    glClear(GL_DEPTH_BUFFER_BIT) # the depth buffer is cleared

    # properties of the object are specified
    # draw red teapot
    glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
    glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0])
    glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0])
    glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)
    #glutSolidTeapot(size)       #generates a solid teapot model of relative size size
    glutSolidTeapot = platform.createBaseFunction( 
    'glutSolidTeapot', dll=platform.PLATFORM.GLUT, resultType=None, 
    argTypes=[GLdouble],
    doc='glutSolidTeapot( GLdouble(size) ) -> None', 
    argNames=('size',)
)

def load_and_draw_model(filename):
    """ Loads a model from an .obj file using objloader.py.
    Assumes there is a .mtl material file with the same name. """
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)

    # set model color
    glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
    glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.75,1.0,0.0])
    glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)
    
    # load from a file
    import objloader
    obj = objloader.OBJ(filename,swapyz=True)
    glCallList(obj.gl_list)

def setup():
    """ Setup window and pygame environment. """
    pygame.init()
    pygame.display.set_mode((width,height),OPENGL | DOUBLEBUF)
    pygame.display.set_caption('OpenGL AR demo')

# load camera data
with open('ar_camera.pkl','r') as f:
    K = pickle.load(f)
    Rt = pickle.load(f)

setup()
draw_background('./Test-Images/book_perspective.bmp')
set_projection_from_camera(K)
set_modelview_from_camera(Rt)
#load_and_draw_model('./Examples/Toy_Plane/toyplane.obj')
draw_teapot(0.02)

while True:
    event = pygame.event.poll()
    if event.type in (QUIT,KEYDOWN):
        break
    pygame.display.flip()  # draws the object on the screen