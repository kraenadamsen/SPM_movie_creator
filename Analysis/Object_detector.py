
"""

Object detection model

"""

import numpy as np
import cv2
import pygame
import pygame_widgets as pw
from skimage.feature import peak_local_max,hog
import pickle
from sklearn.svm import SVC
import progressbar 

class Object_detector:

    def __init__(self,Data):
  
        self.Data = Data
        self.Frame = Data[:,:,0] 
        self.Frame_corrected = self.Frame
        self.number_of_frames = self.Data.shape[2]
        self.number_of_elements = self.Frame.shape[0]*self.Frame.shape[1]
        self.disp_data = np.empty((512,512, 3), dtype=np.uint8)
        self.points_max = np.empty([])

        ## load pretrained SVM model
        
        self.SVM_model_pretrained = pickle.load(open('Circular_object_detection/SVM_circle_model.sav', 'rb'))
        
        
        ## custom SVM model
        
        self.svc_custom = SVC(verbose = True,kernel = 'poly',degree = 2)
        
        self.custom_features = np.empty([0,96])
        self.custom_Labels = np.empty([0])
        self.Trained = False
        self.Train_cmd = False
        
        
    def RUN (self):
 
        pygame.init()
        
        self.setup()
        
        pygame.display.update()
        
        self.Set_done_cmd(True)
        
        
        while self.done_cmd:
            
            pygame.display.update()
            events = pygame.event.get()
            
            self.button_done.listen(events)
            self.button_done.draw()
            
            self.button_train.listen(events)
            self.button_train.draw()
            
            # updating sliders 
            self.slider_frame_number.listen(events)
            self.slider_frame_number.draw()
            
            self.slider_threshold.listen(events)
            self.slider_threshold.draw()
            
            self.slider_min_dist.listen(events)
            self.slider_min_dist.draw()
            
            self.slider_gauss.listen(events)
            self.slider_gauss.draw()
            
            self.slider_SVM.listen(events)
            self.slider_SVM.draw()
            
            self.slider_custom_SVM.listen(events)
            self.slider_custom_SVM.draw()
            
            self.slider_window_size.listen(events)
            self.slider_window_size.draw()
            
            ## getting values from slides
            
            self.Frame_number = int(self.slider_frame_number.getValue()) 
            
            self.threshold = self.slider_threshold.getValue()
            
            self.min_dist = self.slider_min_dist.getValue()
            
            self.n_gauss = int(self.slider_gauss.getValue())
            
            self.SVM_limit = self.slider_SVM.getValue()
            
            self.custom_SVM_limit = self.slider_custom_SVM.getValue()
            
            self.Window_size = self.slider_window_size.getValue()
            
            ## updating values 
            
            self.text_frame_number.setText(self.Frame_number)
            self.text_frame_number.draw()
            
            self.text_threshold.setText("{:.1f}".format(self.threshold))
            self.text_threshold.draw()
            
            self.text_min_dist.setText(self.min_dist)
            self.text_min_dist.draw()
            
            self.text_gauss.setText(self.n_gauss)
            self.text_gauss.draw()
            
            self.text_SVM.setText("{:.2f}".format(self.SVM_limit))
            self.text_SVM.draw()
            
            self.text_custom_SVM.setText("{:.2f}".format(self.custom_SVM_limit))
            self.text_custom_SVM.draw()

            self.text_window_size.setText(int(self.Window_size))
            self.text_window_size.draw()

            ## updating Frame
            
            self.update_Frame()
            
            self.update_Frame_corrected ()
            
            self.gauss_smooth()
            
            # %% Detection Layers
            
            ## maxima - layer
            self.find_local_max()
            
            ## SVM - layer 
            
            # Get subframes
            
            self.get_all_sub_frames ()
            
            # Inference
                 
            self.SVM_pretrained_inference()    
            
            
            ## custom SVM layer
            
            if self.Trained:
                ## inference
                self.SVM_custom_inference()
            
            if self.Train_cmd:
                self.Train_custum_SVM()
                self.Set_train_cmd(False)
                self.Trained = True
                
            
            
            
            ## Update display
            
            self.points = self.points_max            
            
            self.update_display ()            
 
            
        pygame.display.quit()
        
    def update_display (self):
        
        self.update_disp_image_data()
        
        self.Draw_points ()
        
        
               
    
        self.px = pygame.surfarray.make_surface(self.disp_data)
        
        self.screen.blit(self.px, self.px.get_rect())    
        
    def update_Frame (self):
        
        self.Frame = self.Data[:,:,self.Frame_number]        
        
    def update_Frame_corrected (self):             
        
        self.Frame_corrected = self.Frame 
        
        
    def find_local_max (self):

        self.points_max = peak_local_max(self.Frame, threshold_rel = self.threshold ,min_distance=self.min_dist)
        
    def Draw_points (self):
        
        scale_para = 512 / self.Frame.shape[0]
        
        
        points_ = self.points *scale_para
        
        overlay = self.disp_data.copy()
        for point in points_: 
            
            overlay = cv2.circle(overlay, (int(point[0]),int(point[1])), 4,  color = (0,0,255), thickness = -1)
        
        alpha = .3
        self.disp_data = cv2.addWeighted(overlay, alpha, self.disp_data, 1 - alpha, 0)
        
    def gauss_smooth (self):
        
        filtered_image = self.Frame_corrected
        
        for i in range (0,self.n_gauss):
        
            kernel = 1/16 * np.array ([[1,2,1],[2,4,2],[1,2,1]])
            
            filtered_image = cv2.filter2D(filtered_image,-1,kernel)
            
        
        self.Frame_corrected = filtered_image
        
        
    def get_frame_from_point (self,Point):
        
        w , h = self.Frame_corrected.shape
        
        
        size = self.Window_size
        
        idx_min =int(Point[0] - size / 2)
        idx_max =int(Point[0] + size / 2)
        idy_min =int(Point[1] - size / 2)
        idy_max =int(Point[1] + size / 2)
    
        sub_frame =  np.zeros((size, size), dtype=np.int16)
          
        if idx_min > 0 and idy_min > 0 and idx_max < w and idy_max  < h:
            
            sub_frame [:,:]           = self.Frame_corrected[idx_min:idx_max,idy_min:idy_max]
            
        elif idx_min < 0 and idy_min > 0 and idx_max  < w and idy_max  < h: # if x val is to small
        
            sub_frame [-idx_min:,:]   = self.Frame_corrected[:idx_max,idy_min:idy_max]
         
        elif idx_min > 0 and idy_min < 0 and idx_max  < w and idy_max  < h: # if y val is to small
    
            sub_frame [:,-idy_min:]   = self.Frame_corrected[idx_min:idx_max,:idy_max]
    
        elif idx_min > 0 and idy_min > 0 and idx_max  > w and idy_max  < h: # if x val is to large
        
            sub_frame [: w -idx_min,:] = self.Frame_corrected[idx_min:,idy_min:idy_max]
            
            
        elif idx_min > 0 and idy_min > 0 and idx_max  < w and idy_max  > h: # if y val is to large
        
            sub_frame [:,:h-idy_min]   = self.Frame_corrected[idx_min:idx_max,idy_min:]
    
        else:
            sub_frame =  np.ones((size, size), dtype=np.int16)
            
        # sub_frame = sub_frame /np.max(sub_frame)
        # sub_frame.astype(np.float64)
        return  sub_frame
    
    def resize_sub_frame (self,subframe):
        return cv2.resize(subframe,(32,32))
        
       
    
    def get_all_sub_frames (self):
        
        self.sub_arrays = list(map(self.get_frame_from_point,self.points_max))

        self.sub_arrays = list(map(self.resize_sub_frame,self.sub_arrays))        

    def Feature_exstraction (self,sub_frame):
        
        feature_inference = hog(sub_frame,orientations=6,pixels_per_cell=(8,8),cells_per_block=(4,4),block_norm= 'L2')
        
        return feature_inference
        
        
    def SVM_pretrained_inference(self):
        
        Features = list(map(self.Feature_exstraction ,self.sub_arrays))
        
        if len(Features):
        
            results = self.SVM_model_pretrained.decision_function(Features)
            
            results = 1-(1/(1+np.exp(results)))
            
            self.results = results > self.SVM_limit
            
            self.points_max = self.points_max[self.results]
        
        
        
    def SVM_custom_inference(self):
        
        self.get_all_sub_frames()
        
        Features = list(map(self.Feature_exstraction ,self.sub_arrays))
        
        
        if len(Features):
            results = self.svc_custom.decision_function(Features)
            
            results = 1-(1/(1+np.exp(results)))
            
            self.results = results > self.custom_SVM_limit
            
            self.points_max = self.points_max[self.results]
        
        
    def Train_custum_SVM(self):
        
        Features = np.array(list(map(self.Feature_exstraction ,self.sub_arrays)))
        
        print(Features.shape)
        
        self.custom_Labels   = np.concatenate([self.custom_Labels,self.results],axis = 0)
        
        self.custom_features = np.concatenate([self.custom_features,Features]  ,axis = 0)
        
        
        self.svc_custom.fit(self.custom_features, self.custom_Labels)

        print(self.svc_custom.score(self.custom_features, self.custom_Labels))

    

    
    def svm_inference(test_image):
        
        feature_inference = hog(test_image,orientations=6,pixels_per_cell=(8,8),cells_per_block=(4,4),block_norm= 'L2').astype(np.float64)
         
        result = SVM_model.decision_function([feature_inference])
        
        return 1-(1/(1+np.exp(result)))
     

    def setup(self):

        ## setups images
  
        self.update_disp_image_data()


        self.px = pygame.surfarray.make_surface(self.disp_data)
    
        image_size = self.px.get_rect()[2:]
   
        Window_size = image_size 
   
        
        Window_size [0] = Window_size [0] + 400
        Window_size [1] = Window_size [1] + 5    
     
    
        self.screen = pygame.display.set_mode( Window_size )
        self.screen.fill((128,128,128))
        self.screen.blit(self.px, self.px.get_rect())
    
    
     
   
        self.button_done = pw.Button(
                    self.screen, Window_size [0] - 400 + 50, 5 , 100, 50, text='Done',
                    fontSize=22, margin=10,
                    inactiveColour=(255, 0, 0),
                    pressedColour=(0, 255, 0), radius=20,
                    onClick=lambda: self.Set_done_cmd(False) )
        self.button_done.draw()
        
   
        self.slider_threshold= pw.Slider(
                self.screen, Window_size [0] - 400 + 25, 110, 150, 10, min = 0, max = 1, step = .05, initial = 0.5) 
        self.slider_threshold.draw()
        
        self.text_threshold  = pw.TextBox(
                self.screen, Window_size [0] - 400 + 0, 60, 50, 40, fontSize=24)
        self.text_threshold.draw()
        
        text_threshold_describe = pw.TextBox(
                self.screen, Window_size [0] - 400 + 60, 50, 60, 40, fontSize=16,
                borderColour=(128, 128, 128),colour=(128, 128, 128))
        text_threshold_describe.setText('Threshold')
        text_threshold_describe.draw()
  
   
        self.slider_min_dist= pw.Slider(
                self.screen, Window_size [0] - 400 + 25, 180, 150, 10, min = 1, max = 100, step = 1,initial = 20) 
        self.slider_min_dist.draw()
        
        self.text_min_dist  = pw.TextBox(
                self.screen, Window_size [0] - 400 + 0, 130, 50, 40, fontSize=24)
        self.text_min_dist.draw()
        
        text_min_dist_describe = pw.TextBox(
                self.screen, Window_size [0] - 400 + 60, 120, 60, 40, fontSize=16,
                borderColour=(128, 128, 128),colour=(128, 128, 128))
        text_min_dist_describe.setText('Minimum Distance ')
        text_min_dist_describe.draw()
        
        
        
        self.slider_SVM= pw.Slider(
                self.screen, Window_size [0] - 400 + 225, 60, 150, 10, min = .05, max = 1, step = 0.01 ,initial = 0.5) 
        self.slider_SVM.draw()
        
        self.text_SVM  = pw.TextBox(
                self.screen, Window_size [0] - 400 + 200, 10, 60, 40, fontSize=24)
        self.text_SVM.draw()
        
        text_SVM_describe = pw.TextBox(
                self.screen, Window_size [0] - 400 + 260, 0, 100, 40, fontSize=16,
                borderColour=(128, 128, 128),colour=(128, 128, 128))
        text_SVM_describe.setText('SVM limit')
        text_SVM_describe.draw()
        
        
        self.slider_gauss= pw.Slider(
                self.screen, Window_size [0] - 400 + 25, 320, 150, 10, min = 0, max = 10, step = 1,initial = 1) 
        self.slider_gauss.draw()
        
        self.text_gauss  = pw.TextBox(
                self.screen, Window_size [0] - 400 + 0, 270, 60, 40, fontSize=24)
        self.text_gauss.draw()
        
        text_gauss_describe = pw.TextBox(
                self.screen, Window_size [0] - 400 + 60, 260, 100, 40, fontSize=16,
                borderColour=(128, 128, 128),colour=(128, 128, 128))
        text_gauss_describe.setText('# Gauss')
        text_gauss_describe.draw()
        
        
        
        self.slider_frame_number = pw.Slider(
                self.screen, Window_size [0] - 400 + 25, 450, 150, 10, min = 1, max = self.number_of_frames-1, step = 1,initial = 0)
        self.slider_frame_number.draw()
        
        self.text_frame_number  = pw.TextBox(
                self.screen, Window_size [0] - 400 + 0, 400, 60, 40, fontSize=24)
        self.text_frame_number.draw()
        text_frame_number_describe = pw.TextBox(
                self.screen, Window_size [0] - 400 + 60, 390, 100, 40, fontSize=16,
                borderColour=(128, 128, 128),colour=(128, 128, 128))
        text_frame_number_describe.setText('Frame Number')
        text_frame_number_describe.draw() 
        
        
        
        self.button_train = pw.Button(
            self.screen, Window_size [0] - 400 + 250, 100 , 100, 50, text='Train',
            fontSize=22, margin=10,
            inactiveColour=(255, 0, 0),
            pressedColour=(0, 255, 0), radius=20,
            onClick=lambda: self.Set_train_cmd(True) )
        self.button_train.draw()
        
        
        self.slider_custom_SVM = pw.Slider(
                self.screen, Window_size [0] - 400 + 225, 210, 150, 10, min = .05, max = 1, step = 0.02 ,initial = 0.5) 
        self.slider_custom_SVM.draw()
        
        self.text_custom_SVM  = pw.TextBox(
                self.screen, Window_size [0] - 400 + 200, 160, 60, 40, fontSize=24)
        self.text_custom_SVM.draw()
        
        text_custom_SVM_describe = pw.TextBox(
                self.screen, Window_size [0] - 400 + 260, 170, 100, 40, fontSize=16,
                borderColour=(128, 128, 128),colour=(128, 128, 128))
        text_custom_SVM_describe.setText('Custom SVM limit')
        text_custom_SVM_describe.draw()
        
        
        self.slider_window_size = pw.Slider(
            self.screen, Window_size [0] - 400 + 225    ,300, 150, 10, min = 8, max = 48, step = 1 ,initial = 32) 
        self.slider_window_size.draw()
        
        self.text_window_size  = pw.TextBox(
                self.screen, Window_size [0] - 400 + 200, 250, 60, 40, fontSize=24)
        self.text_window_size.draw()
        
        text_window_size_describe = pw.TextBox(
                self.screen, Window_size [0] - 400 + 260, 260, 100, 40, fontSize=16,
                borderColour=(128, 128, 128),colour=(128, 128, 128))
        text_window_size_describe.setText('window size')
        text_window_size_describe.draw()
        
        
        
    def update_disp_image_data(self):
        
        im = self.Frame_corrected 
        
        im = cv2.resize(im,(512,512))
       
        im   = (im - (np.min(im))) 
        im = 255 * (im / np.max(im))
       
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        im = clahe.apply(im.astype(np.uint8))
       
        im = np.transpose(im)    
        
        self.disp_data[:, :, 2] = im
        self.disp_data[:, :, 1] = im
        self.disp_data[:, :, 0] = im
        
        
    def Set_done_cmd(self,cmd):
        
        self.done_cmd = cmd
        
    def Set_train_cmd(self,cmd):
        
        self.Train_cmd = cmd
        
        
    #%% Get data after creation
    
    
    
    def Get_points (self):
        
        return_dict = {}
        
        widgets_get_points = ['[', 
          progressbar.Timer(format= 'Getting points time:            %(elapsed)s'), 
          '] ', 
            progressbar.Bar('*'),' (', 
            progressbar.ETA(), ') '] 

        bar_points = progressbar.ProgressBar(max_value=self.number_of_frames,widgets=widgets_get_points).start()
        
        for self.Frame_number in range(0,self.number_of_frames):
            
            self.update_Frame()
            
            self.update_Frame_corrected ()
            
            self.gauss_smooth()
            
            # %% Detection Layers
            
            ## maxima - layer
            self.find_local_max()
            
            ## SVM - layer 
            
            # Get subframes
            
            self.get_all_sub_frames ()
            
            # Inference
                 
            self.SVM_pretrained_inference()    
            
            
            ## custom SVM layer
            
            if self.Trained:
                self.SVM_custom_inference()
                
                
            self.points = self.points_max 
                
            return_dict[str(self.Frame_number)] = self.points
            
            bar_points.update(self.Frame_number)
                
                
        return return_dict
            

    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        