"""

Data treatment

"""
import matplotlib.pylab as pl
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import Object_detector
import Kmean_model
import ctypes
from ctypes import cdll
from scipy import fftpack, ndimage,signal
from pystackreg import StackReg
import progressbar 
from skimage.feature import peak_local_max,hog
import os
# import compiled_functions

file_local_dir = os.path.dirname(__file__)

lib = cdll.LoadLibrary(file_local_dir+'/C_functions/HHCF/HHCF.dll' )


# %% import Data
hf = h5py.File('../Saved_data/Non_Aligned_data.h5', 'r')
Data = np.array(hf.get('Data'))
hf.close()


#%% ################ image segmentation using Kmeans #########################

my_model =  Kmean_model.Kmeans_model(Data)

my_model.RUN()

all_labels = my_model.get_all_labels()




# # %% counts ratio between "layers"



# counts_all = np.empty([my_model.n,my_model.number_of_frames])

# for i in range (0,my_model.number_of_frames):
#     unique, counts = np.unique(all_labels[:,:,i], return_counts=True)
#     counts_all [:,i] = counts


# ## plotting counts and fit

# plt.figure()

# frame_rate = 4
# time = np.arange(my_model.number_of_frames) * frame_rate
# fit =np.polyfit(time,counts_all[0,:]/(512*512),deg = 1)
# plt.plot(time,counts_all[0,:]/(512*512),'.r')
# plt.plot(time,fit[0]*time+fit[1],'r-')

# fit = np.polyfit(time,counts_all[1,:]/(512*512),deg = 1)
# plt.plot(time,counts_all[1,:]/(512*512),'.b')
# plt.plot(time,fit[0]*time+fit[1],'b-')

# fit = np.polyfit(time,counts_all[2,:]/(512*512),deg = 1)
# plt.plot(time,counts_all[2,:]/(512*512),'.g')
# plt.plot(time,fit[0]*time+fit[1],'g-')


# %% ############################ HHCF calculations ##########################

######## python solution - very slow #########
# def HHCF (image_np,R,im_size):

#     image = image_np.tolist() 
        
#     image_sum = [0] * R 
  
    

#     for r in range (R):
#         for i in range(0,im_size): # iterates over fast scanning lines
#             line_sum = .0
        
#             for j in range(0,im_size-r): # iterates through fast scanning lines

            
#                 line_sum = line_sum + (image[j + (i * im_size)] - image[j+ (i * im_size) + r])**2
            
#             image_sum[r] = image_sum[r] + (line_sum / (im_size - r))
     
#         image_sum[r] = image_sum[r] / im_size
        
#     return image_sum

# HHCF(Data[:,:,40].flatten(),2,256)


########### C / C++ solution - fast ############


# lib.HHCF.restype = ctypes.c_double
# lib.HHCF.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64), ctypes.c_int, ctypes.c_int]



# widgets_HHCF = [' [', 
#           progressbar.Timer(format= 'Calculation time:   %(elapsed)s'), 
#           '] ', 
#             progressbar.Bar('*'),' (', 
#             progressbar.ETA(), ') '] 
# bar_HHCF = progressbar.ProgressBar(max_value=Data.shape[2],widgets=widgets_HHCF).start() 

# HHCF_results = np.zeros([Data.shape[2],199])
# for i in range(0,Data.shape[2]):
#     test_data = Data [:,:,i]/32.768
#     HHCF_results[i,:] = [lib.HHCF(test_data, test_data.shape[0],r) for r in range(1,200)]
    
#     bar_HHCF.update(i)



# #%%plotting HHCF
# colors = pl.cm.jet(np.linspace(0,1,Data.shape[2]))
# plt.close('all')
# plt.figure()
# points = [0]*Data.shape[2]
# J = 18
# Radius = np.linspace(0, 100*(199/511),199)

# for i in range (0,Data.shape[2]):
#     plt.plot( Radius , np.array(HHCF_results[i,:])/np.max(HHCF_results[i,:]),color = colors[i,:],lw = .5)
#     points [i] = np.array(HHCF_results[i,J])/np.max(HHCF_results[i,:])
    
# # plt.vlines ( Radius [J] , 0,1 )
# # plt.xlim([0,11]) 
# plt.xlabel('radius [nm]')
# plt.ylabel('HHCF [nm^2]')

# plt.figure()

# for i in range (0,Data.shape[2]):
#     plt.plot( Radius , np.array(HHCF_results[i,:])/np.max(HHCF_results[i,:]),color = colors[i,:],lw = .5)
#     points [i] = np.array(HHCF_results[i,J])/np.max(HHCF_results[i,:])
    
# # plt.vlines ( Radius [J] , 0,1 )
# plt.xlim([0,11]) 
# plt.xlabel('radius [nm]')
# plt.ylabel('HHCF [nm^2]')


# plt.figure()

# for i in range (0,Data.shape[2]):
#     plt.plot(Radius ,np.gradient( np.array(HHCF_results[i,:])/np.max(HHCF_results[i,:])),color = colors[i,:],lw =.5)

# plt.xlim([0,11])
# # plt.ylim([-0.003,0.006])
# plt.xlabel('radius [nm]')
# plt.ylabel('Gradient')
 

# plt.figure()

# for i in range (0,Data.shape[2]):
#     plt.plot(Radius ,np.gradient(np.gradient( np.array(HHCF_results[i,:])/np.max(HHCF_results[i,:]))),color = colors[i,:],lw =.5)

# plt.xlim([0,11])
# plt.ylim([-0.003,0.006])
# plt.xlabel('radius [nm]')
# plt.ylabel('Curvature')



# #%% #########################   object detection ###########################


# my_model = Object_detector.Object_detector(Data)
# my_model.RUN()

# #%%

# points = my_model.Get_points()


# # %% diffusion analysis

# #%%
# NN_dist = []

# widgets_calc = ['[', 
#           progressbar.Timer(format= 'Calculation time:            %(elapsed)s'), 
#           '] ', 
#             progressbar.Bar('*'),' (', 
#             progressbar.ETA(), ') '] 

# bar_calc = progressbar.ProgressBar(max_value=Data.shape[2]-1,widgets=widgets_calc).start()

# sr = StackReg(StackReg.TRANSLATION)
# for i in range (0,Data.shape[2]-1):
    
#     sr.register(Data[:,:,i] , Data[:,:,i+1])
#     tranlation_matrix = sr.get_matrix()
    
#     points_First = [[points[str(i)][x][0],points[str(i)][x][1]] for x in range (0,len(points[str(i)]))]
    
#     points_Second = [[points[str(i+1)][x][0]-tranlation_matrix[0,-1],points[str(i+1)][x][1]-tranlation_matrix[1,-1]] for x in range (0,len(points[str(i+1)]))]
    
#     for point in points_First:
#         dist_x = (point[0]- np.array(points_Second)[:,0])
#         dist_y = (point[1]- np.array(points_Second)[:,1])
#         DIST = np.sqrt(dist_x**2 + dist_y**2)
#         NN_dist.append(np.min(DIST))
        
#     bar_calc.update(i)
        

# print('\n')  
# #%%

# plt.hist(NN_dist,bins = np.linspace(0,20/(300/256),20))


# #%% calculate rates

# def moved (x):
#     return x>5 and x<10
    

# O = 0
# rate = []

# for i in range (0,Data.shape[2]-1):
    
#     number_of_points_in_frame = len(points[str(i)])
    
#     number_of_moved_molecules =  len(list(filter(moved ,NN_dist [O:O + number_of_points_in_frame])))
    
    
#     if number_of_moved_molecules / number_of_points_in_frame > 0.15:
#         rate.append(np.nan)
#     else:
            
#         rate.append(number_of_moved_molecules / number_of_points_in_frame)
    
#     O = O + number_of_points_in_frame


        
# plt.figure()  
# plt.plot(rate,'.')
# # plt.yscale('log')
    
    
    

    
    

 









    
    
    
