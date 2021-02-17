
"""
Utilities

Used  in Area selection for tilt correction

"""


import numpy as np
import gwyfile
import cv2
from skimage.exposure import cumulative_distribution
import h5py
from sklearn.cluster import KMeans


#%% plane fit 
def gray(im):
     im   = (im - (np.min(im))) 
     im = 255 * (im / np.max(im))
   
     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
     im = np.clip(im, 0, 255)
     im = im.astype(np.int8) 
     
     # im = clahe.apply(im)
     

     return im


def scale_contrast (M):
    M   = (M + np.abs(np.min(M))) 
    M   = M /np.max(M) * 255 
    return M

## filters

def edge_detection(Frame):
    filter_array = np.array ([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    
    filtered_image = convolve (Frame,filter_array) 
    return filtered_image

def gauss_smooth(Frame):
    filter_array = 1/16 * np.array ([[1,2,1],[2,4,2],[1,2,1]])
    
    filtered_image = convolve (Frame,filter_array) 
    return filtered_image

def mean_smooth(Frame):
    filter_array = 1/9 * np.array ([[1,1,1],[1,1,1],[1,1,1]])
    
    filtered_image = convolve (Frame,filter_array) 
    return filtered_image

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def three_point_corrector(points,M):
    
    
    y0, x0, z0, y1, x1, z1, y2, x2, z2 = points
    
    ux, uy, uz =  [x1-x0, y1-y0, z1-z0]
    vx, vy, vz =  [x2-x0, y2-y0, z2-z0]
    
    u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]
    
    point  = np.array((x0, y0, z0))
    normal = np.array(u_cross_v)
    
    d = -point.dot(normal)
    
    xx, yy = np.meshgrid(range(len(M)), range(len(M)))
    
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2] 

    return M-z   


def get_Data(Data,ref_file_number,ref_frame_number):

    VZ_scale =  Data[ref_file_number][f'IMG_{ref_frame_number:03d}']['Feedback'][0,0][0][0][-1][0]
    AAperVZ  =  Data[ref_file_number][f'IMG_{ref_frame_number:03d}']['Config'][0,0][0][0][-1][0]
    img_data =  Data[ref_file_number][f'IMG_{ref_frame_number:03d}']['Data'][0,0] / 32768 * VZ_scale * AAperVZ
    
    Frame = (img_data - np.mean(img_data))*10**-10 # units in meters
    
    Frame  = Frame [::-1,:]
    Frame = Frame.transpose()
    
    
    return Frame


def set_Data(new_gwy_container,ref_file_number,ref_frame_number,Data,M,frame_num):

    str_number =f'/{frame_num}/data' 
    
    new_gwy_container[str_number] = gwyfile.objects.GwyDataField(M)
    # new_gwy_container[str_number]['xreal']      = Data[ref_file_number]['/%s/data'% ref_frame_number]['xreal']
    # new_gwy_container[str_number]['yreal']      = Data[ref_file_number]['/%s/data'% ref_frame_number]['yreal']
    # new_gwy_container[str_number]['si_unit_xy'] = Data[ref_file_number]['/%s/data'% ref_frame_number]['si_unit_xy']
    # new_gwy_container[str_number]['si_unit_z']  = Data[ref_file_number]['/%s/data'% ref_frame_number]['si_unit_z']
    # new_gwy_container[str_number]['Bias']      = Data[ref_file_number]['/%s/meta'% ref_frame_number]['Bias']
    # new_gwy_container[str_number]['Current']      = Data[ref_file_number]['/%s/meta'% ref_frame_number]['Current']
    # new_gwy_container[str_number]['Scan duration'] = Data[ref_file_number]['/%s/meta'% ref_frame_number]['Scan duration']
    

    return new_gwy_container

def cdf(im):
    '''
    computes the CDF of an image im as 2D numpy ndarray
    '''
    c, b = cumulative_distribution(im) 
    # pad the beginning and ending pixels and their CDF values'

    
    c = np.insert(c, 0, [0]*b[0])
    
    
    # c = np.append(c, [1]*(255-b[-1]))

    c = np.append(c, [1]*(256-len(b)))
    return c


    return im

def Saving_data_to_hdf5_file(path , size , Data ):
    
    hf = h5py.File(path, 'w')
    
    hf.create_dataset ('Data',\
                       shape = size,\
                       data = Data)
        
    hf.close()
    
    return print('file saved')


def from_float_to_16_bit (Data):
    
    def convert (Frame):
        minimum = Frame.min()
        Frame = Frame - minimum
        maximum = Frame.max()
        Frame = (Frame / maximum) * 32767
        Frame = Frame.astype(np.int16)
        return Frame
    
    bit_16_Data = np.empty(Data.shape,dtype=np.int16)
    
    for i in range(0,Data.shape[2]):
         bit_16_Data[:,:,i] = convert (Data[:,:,i])
         
    return bit_16_Data
        


def hist_match (im_temp,im_ref, resolution = 32768):
    
    hist_temp ,bins= np.histogram(im_temp.flatten(),bins = resolution)
    hist_ref ,bins = np.histogram(im_ref.flatten(),bins = resolution)
    
    cdf_temp  = np.cumsum(hist_temp)
    cdf_ref = np.cumsum(hist_ref)
    
    im_ref = (np.floor((resolution-1) * (im_ref / np.max(im_ref)))).astype(np.int)
    
    transformation = np.interp(cdf_ref, cdf_temp, np.arange(resolution))
    
    im_trans = (np.reshape(transformation[im_ref.ravel()], im_ref.shape))
    im_matched = 255 * (im_trans / np.max(im_trans))
    
    return im_matched

def kmeans_calc (Data, n , reduction_parametor = 100):
    
    centres_init = np.linspace(Data.min(),Data.max(),n).reshape(n,1)
    
    
    ## perform Kmeans sacling         
    data_reduced  = Data[::reduction_parametor].flatten()
    data_reduced  = data_reduced.reshape (len(data_reduced),1)
    
    ## calcultaed Kmean model
    kmeans   = KMeans(init = centres_init , n_clusters=n,n_init=1).fit(data_reduced)
    
    return kmeans.cluster_centers_

def truncanate (Frame):
    
    Frame  = np.where(Frame >= 0  , Frame, 0)
    
    Frame  = np.where(Frame <= 255, Frame, 255)
    
    return Frame


def hist_match_kmeans (im_temp , im_ref , n , temp_centers = []):

    
    
    for i in range (0,5):

        im_temp_blur = cv2.blur(im_temp,(3,3))
        im_ref_blur = cv2.blur(im_ref,(3,3))
    
    
        
    
    # %% reference calculations
    
    ref_centers = kmeans_calc (im_ref_blur, n)
    

    
    if temp_centers == []:
        
        temp_centers = kmeans_calc (im_temp_blur, n)
    else:
        
        pass
    
    
    ### sorting centers
    ref_centers  = np.sort(ref_centers)
    temp_centers = np.sort(temp_centers)
    
    # create transformation parameters - simple 1. order correction
    transformation = np.polyfit(ref_centers[:,0], temp_centers[:,0], 1)
    
    im_trans = transformation[0]* im_ref + transformation[1]
    
    # im_trans = (im_trans - (np.min(im_trans)))     
    # im_matched = 255 * (im_trans / np.max(im_trans))
    
    im_matched = truncanate (im_trans)
 
    
    return im_matched
    
    











