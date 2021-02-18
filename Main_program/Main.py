#%% libaries import 

"""
Will run a file, whitch imports all modules needed 

"""
    
f  = open('../Utilities/libaries.py',"r")    
exec(f.read())
f.close()

#%% Data import 

"""
Data import 

This will look for all .mat files in the /Muls/.. folder and import them 

The imported Data is a python list, holding a imported dictionary 

images can be accesed by using utilities.get_Data ( filenumber, frame_number)

"""
filenames = glob.glob('../Muls/*.mat')
        
Data = []



widgets_import = [' [', 
         progressbar.Timer(format= 'Import time:              %(elapsed)s'), 
         '] ', 
           progressbar.Bar('*'),' (', 
           progressbar.ETA(), ') '] 
bar_import = progressbar.ProgressBar(max_value=len(filenames),widgets=widgets_import).start() 

for file in filenames:
        obj = scipy.io.loadmat(file)
        bar_import.update(filenames.index(file))
        
        Data.append(obj)
print('\n')        
# %% selecting frame

"""
Selection frames to be used in the movie  - opens a GUI interphase

Creates a python dictionary which holds references runnning from 1 to the number frames in the movie
with keys pointing to the correct file in the Data variable using utilities.get_Data 

"""

Refence_dict = Selecting_frames_for_movie.run(Data)


        
# %%  plane correction and contrast

"""
Lets you choose 3 points to level all images and adjust constrast and brightness of an image to wiich others will be alligned.

return 3 points values to correct all frames using utilities.three_point_corrector(level_correction_val,Frame)

and returns the image to which contrast and bightness will be alligned.

"""


(level_correction_val , contrast_adjusted_image) = leveling_contrast.run(Data, Refence_dict, Frame_used_for_correction = 1)


# %% level all images and creates new data holders.

"""
Initiate new data containers, which are saved in the end.  
Including 1 gwyddion file and two Numpy arrays: 1 for the spatial alligned images and on for the non alligned.
 
"""
w,h =  utilities.get_Data(Data,Refence_dict[1][0],Refence_dict[1][1]).shape

new_gwy_container = gwyfile.objects.GwyContainer()

Non_aligned_Data= np.zeros ([w,h,len(Refence_dict)])

Aligned_Data = np.zeros ([w,h,len(Refence_dict)])

## creating a progress bar
widgets_level = ['[', 
          progressbar.Timer(format= 'Leveling time:            %(elapsed)s'), 
          '] ', 
            progressbar.Bar('*'),' (', 
            progressbar.ETA(), ') '] 

bar_level = progressbar.ProgressBar(max_value=len(Refence_dict),widgets=widgets_level).start()

## iterates of all frames in movie and levels them 

for image_ref in Refence_dict.keys():

    M = utilities.get_Data(Data,Refence_dict[image_ref][0],Refence_dict[image_ref][1])
    
    M_corrected =  utilities.three_point_corrector(level_correction_val,M)
    
    
    ################### Include filters here id needed #######################
    
    
    # ## add filter functions if needed here:
    # # like this:
    # # M_corrected = utilities.gauss_smooth(M_corrected)
    
    new_gwy_container = utilities.set_Data(new_gwy_container,
              Refence_dict[image_ref][0],Refence_dict[image_ref][1],
              Data,M_corrected,image_ref )
    
    
    # transpose to show as image
    Non_aligned_Data [:,:,image_ref-1] = M_corrected.transpose() 
    
    # updates progress bar
    bar_level.update(image_ref)


# %% Matches contrast, brightness and correctes drift


widgets_contrast = [' [', 
          progressbar.Timer(format= 'Constrast alignment time: %(elapsed)s'), 
          '] ', 
            progressbar.Bar('*'),' (', 
            progressbar.ETA(), ') '] 
  
bar_contrast = progressbar.ProgressBar(max_value=len(Refence_dict),widgets=widgets_contrast).start()

###################### select type of drift compensation #####################

# sr = StackReg(StackReg.RIGID_BODY)
sr = StackReg(StackReg.TRANSLATION)


for image_ref in Refence_dict.keys():
    
    # gets data
    
    im = Non_aligned_Data [:,:,image_ref-1].transpose()

    # transforms to image format
    im = (im - (np.min(im))) 
    im = 255 * (im / np.max(im))
    
    ################### select type of contrast matching #####################
    # im = utilities.hist_match(contrast_adjusted_image,im)
    im = utilities.hist_match_kmeans(contrast_adjusted_image,im, n = 3)

    # saves un aligned image

    cv2.imwrite(f'../Image_frames/im_{image_ref}.png',(im.transpose()).astype(np.uint8))
    
    # aligns image positions
    
    sr.register(contrast_adjusted_image , im)
    tranlation_matrix = sr.get_matrix()
    im = sr.transform(im)
        
    # translates 
    
    Aligned_Data [:,:,image_ref-1] = sr.transform(Non_aligned_Data [:,:,image_ref-1]).transpose()
    
    # Saves aligned images 
    
    cv2.imwrite(f'../Image_frames_aligned/im_{image_ref}.png',(im.transpose()).astype(np.uint8))
    
    bar_contrast.update(image_ref)


# %% save tiltcorrected data

print('\n Saving New Data files')

new_gwy_container.tofile('../Saved_data/new_file.gwy')

# %% save hdf5 files 

# convert to 16 bit int for compression 

Non_aligned_Data = utilities.from_float_to_16_bit(Non_aligned_Data)

Aligned_Data = utilities.from_float_to_16_bit(Aligned_Data)

# saving

utilities.Saving_data_to_hdf5_file (path = '../Saved_data/Non_aligned_data.h5',\
                          size = (w,h,len(Refence_dict)),\
                          Data =  Non_aligned_Data)

utilities.Saving_data_to_hdf5_file (path = '../Saved_data/Aligned_data.h5',\
                          size = (len(contrast_adjusted_image),len(contrast_adjusted_image),len(Refence_dict)),\
                          Data =  Aligned_Data)


# %% Finished

print('\n Finished')

