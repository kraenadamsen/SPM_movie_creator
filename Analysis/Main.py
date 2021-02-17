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

lib = cdll.LoadLibrary('HCCF_calculator/x64/Debug/HCCF_calculator.dll')


hf = h5py.File('../Saved_data/Non_Aligned_data.h5', 'r')
Data = np.array(hf.get('Data'))
hf.close()


# AC = signal.fftconvolve(Data[:,:,5],Data[:,:,5].transpose())

# #%%

# cv2.imshow('auto coor',AC/AC.max())

#%%

# test_data = Data[:,:,1]/32768.0

# cv2.imshow('test',test_data)

# lib.HHCF.restype = ctypes.c_double
# lib.HHCF.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),	ctypes.c_int]
# HHCF_results = [lib.HHCF(test_data, r) for r in range(1,200)]

# Radius = np.linspace(1, 15*(199/512),199)
# plt.plot(Radius,HHCF_results)






# # %% HHCF calculations 

# lib.HHCF.restype = ctypes.c_double
# lib.HHCF.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),	ctypes.c_int]


# HHCF_results = np.zeros([Data.shape[2],199])
# for i in range(0,Data.shape[2]):
#     test_data = Data [:,:,i]/32.768
#     HHCF_results[i,:] = [lib.HHCF(test_data, r) for r in range(1,200)]

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

# plt.figure()
# plt.plot(points)




#%%
cv2.imwrite ('test.png',Data[:,:,29]/Data[:,:,29].max()*255)



filtered_image = Data[:,:,29]

for i in range (0,10):

    filtered_image = cv2.blur(filtered_image,(3,3))
    
cv2.imwrite ('blur.png',filtered_image/Data[:,:,29].max()*255)

# my_model =  Kmean_model.Kmeans_model(Data)
# my_model.RUN()

# all_labels = my_model.get_all_labels()

# # %%

# cv2.imshow('ha',all_labels[:,:,16]/all_labels[:,:,16].max())

# counts_all = np.empty([my_model.n,my_model.number_of_frames])
# for i in range (0,my_model.number_of_frames):
#     unique, counts = np.unique(all_labels[:,:,i], return_counts=True)
#     counts_all [:,i] = counts

# plt.close('all')
# plt.figure()
# for i in range (0,my_model.n):
#     plt.plot(counts_all[i,:]/(512*512),'.')
    

# plt.figure()

# frame_rate = 4
# time = np.arange(my_model.number_of_frames) * frame_rate
# fit =np.polyfit(time,counts_all[0,:],deg = 1)
# plt.plot(time,counts_all[0,:],'.r')
# plt.plot(time,fit[0]*time+fit[1],'r-')

# fit =np.polyfit(time,counts_all[1,:],deg = 1)
# plt.plot(time,counts_all[1,:],'.b')
# plt.plot(time,fit[0]*time+fit[1],'b-')

# fit =np.polyfit(time,counts_all[2,:],deg = 1)
# plt.plot(time,counts_all[2,:],'.g')
# plt.plot(time,fit[0]*time+fit[1],'g-')



#%% object detection


my_model = Object_detector.Object_detector(Data)
my_model.RUN()

#%%
# n =73

# n = 54
# n = 4 

# cv2.imshow('NH_3',cv2.resize((my_model.sub_arrays[n]-my_model.sub_arrays[n].min())/my_model.sub_arrays[n].max(),(256,256)))
# cv2.imwrite('NH_3.png',cv2.resize((my_model.sub_arrays[n]-my_model.sub_arrays[n].min())/my_model.sub_arrays[n].max()*300,(256,256)))

# fd,hog_img = hog(cv2.resize((my_model.sub_arrays[n]-my_model.sub_arrays[n].min())/my_model.sub_arrays[n].max()*300,(256,256)),orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True)
                 
# cv2.imwrite('HOG_NH3.png',hog_img/hog_img.max()*255)

# #%%
# plt.plot(fd)


# # %% 



# points = my_model.Get_points()

# %%

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
    
    
#     if number_of_moved_molecules / number_of_points_in_frame > 0.3:
#         rate.append(np.nan)
#     else:
            
#         rate.append(number_of_moved_molecules / number_of_points_in_frame)
    
#     O = O + number_of_points_in_frame


        
# plt.figure()  
# plt.plot(rate,'.')
# plt.yscale('log')
    
    
    

    
    

 









    
    
    
