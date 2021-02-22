"""

Find spherical particle 

"""

import random
import cv2
import numpy as np

from scipy import ndimage

from sklearn.model_selection import train_test_split

from sklearn.calibration import CalibratedClassifierCV

import numpy as np

image_size = 32

## Gererating Data

# define normalized 2D gaussian
def gaus2d(x=0, y=0, mx=0, my=0, sx=.8, sy=1):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

x = np.linspace(-2, 2,image_size)
y = np.linspace(-2, 2,image_size )
x, y = np.meshgrid(x, y) # get 2D variables instead of 1D

number_of_samples = 20000

Generated_data_True = np.empty([number_of_samples,image_size ,image_size ])
Generated_data_False = np.empty([number_of_samples,image_size ,image_size ])
Generated_data_False_line_strike = np.empty([number_of_samples,image_size ,image_size ])

for i in range (0,number_of_samples):
    
    mx = random.randint(-5, 5)/7
    my = random.randint(-5, 5)/7
    
    sx = np.random.normal(loc = .8, scale = .2)
    sy = np.random.normal(loc = .8, scale = .2)
    
    z = gaus2d(x, y, mx, my, sx)
    
    theta = np.radians(random.randint(0,360))

    z = ndimage.rotate(z, theta, reshape=False)
    
    noise = np.random.normal(0, .01, z.shape)
    z = z + noise
    
    
    
    
    Generated_data_True[i,:,:] = z/z.max()
    
    Generated_data_False[i,:,:] = np.random.normal(0, .3, z.shape)/z.max()
    
    if i%2:
    
        Generated_data_False_line_strike[i,:,:] = np.random.normal(0, .3, z.shape)/z.max()
        
        row = random.randint(12,22)
        
        width = random.randint(1,8)

        
        Generated_data_False_line_strike[i,row : row + width,:] = np.ones([width,image_size ])
    else:
        
        Generated_data_False_line_strike[i,:,:] = z/z.max()
        
        row = random.randint(12,22)
        
        width = random.randint(1,8)
        
        
        
        Generated_data_False_line_strike[i,row : row + width,:] = np.ones([width,image_size ])
    
    
    

labels =np.asarray( np.concatenate([np.ones(number_of_samples),np.zeros(number_of_samples),np.zeros(number_of_samples)]))

# labels = tf.keras.utils.to_categorical(labels, num_classes=2)

Train_data = np.concatenate([Generated_data_True,Generated_data_False,Generated_data_False_line_strike],axis = 0)

print(Train_data.shape)


#%% preparedata for CNN

def shuffle(a,b):
    
    arr = np.arange(a.shape[0])
    np.random.shuffle(arr)

    return (a[arr,:,:],b[arr])

(Train_data,labels)  = shuffle(Train_data,labels)

Train_data = np.expand_dims(Train_data, axis=-1)

x_train, x_test, y_train, y_test = train_test_split(Train_data, labels, test_size = 0.2)




# %% SVM 

print('SCV test')

from sklearn.svm import LinearSVC
from skimage.feature import hog

features = np.empty ([x_train.shape[0],96])
features_test = np.empty ([x_test.shape[0],96])

for i in range (0,x_train.shape[0]):

    features[i,:]  = hog(x_train[i,:,:,:],orientations=6,pixels_per_cell=(8,8),cells_per_block=(4,4),block_norm= 'L2')  

for i in range (0,x_test.shape[0]):

    features_test[i,:] = hog(x_test[i,:,:,:],orientations=6,pixels_per_cell=(8,8),cells_per_block=(4,4),block_norm= 'L2')  
    
    
svc = LinearSVC(verbose = True)

# svc = CalibratedClassifierCV(svc) 


svc.fit(features, y_train)


accuracy = svc.score(features_test, y_test)
print(accuracy)





#%% inference test
 
import time
import pickle
image_test = x_test [650,:,:,:]

 
cv2.imshow('test_image_false',cv2.resize(image_test,(256,256)))
cv2.imwrite('test_image_false.png',cv2.resize(image_test,(256,256))*255)

def svm_inference(test_image):
    
    feature_inference = hog(test_image,orientations=6,pixels_per_cell=(8,8),cells_per_block=(4,4),block_norm= 'L2')
     
    result = svc.decision_function([feature_inference])
    
    return 1/(1+np.exp(result)) 


print(svm_inference(image_test))


start = time.time()
for i in range(0,100):
    svm_inference(x_test [i,:,:,:])
    
print('SVM inference time :',time.time()-start)




filename = 'SVM_circle_model.sav'
pickle.dump(svc, open(filename, 'wb'))





