
"""

leveling adn kmeans analysis OOP

"""

import pygame
import numpy as np 
import pygame_widgets as pw
import cv2
from sklearn.cluster import KMeans



class Kmeans_model:
    

    
    def __init__ (self,Data):
        
        self.Data = Data
        self.Frame = Data[:,:,0] 
        self.Frame_corrected = self.Frame
        self.number_of_frames = self.Data.shape[2]
        self.number_of_elements = self.Frame.shape[0]*self.Frame.shape[1]
        self.n = 3        
        self.centers = self.init_centers()
        self.Labels = np.empty(self.Frame.shape,dtype = np.int16)
        self.disp_data = np.empty((self.Frame.shape[0],self.Frame.shape[0], 3), dtype=np.uint8)
        self.disp_Label = np.empty((self.disp_data.shape), dtype=np.uint8)
        
        self.x_slope = 0
        self.y_slope = 0
        
        # initial label setting
        self.calulate_K_mean()
        
        
    # %% main run function

        
        
    # %% k-mean related methods 
    
    def init_centers(self):        
         return np.linspace(self.Frame_corrected.min(),self.Frame_corrected.max(),self.n).reshape(self.n,1)
        
        
    def calulate_K_mean(self,reduction_parametor = 100):
           
        ## perform Kmeans sacling         
        data_reduced  = self.Frame_corrected[::reduction_parametor].flatten()
        data_reduced  = data_reduced.reshape (len(data_reduced),1)
        
        ## calcultaed Kmean model
        kmeans   = KMeans(init = self.centers , n_clusters=self.n,n_init=1).fit(data_reduced)
        
        ## Prediction label to all points         
        results = kmeans.predict(self.Frame_corrected.flatten().reshape(self.number_of_elements,1))

        
        ## updates center - good speed up trick!
        self.centers = kmeans.cluster_centers_
        
        
        
        ## creates labels image 
        self.Labels = results.reshape(self.Frame.shape[0],self.Frame.shape[1])
        
    #%% pygame  related methods
     
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
            
            self.slider_x.listen(events)
            self.slider_x.draw() 
            
            self.slider_y.listen(events)
            self.slider_y.draw()
            
            
            self.slider_frame_number.listen(events)
            self.slider_frame_number.draw()
            
            self.slider_n.listen(events)
            self.slider_n.draw()
            
            self.slider_gauss.listen(events)
            self.slider_gauss.draw()
            
            # get slider values
            
            self.y_slope = self.slider_y.getValue()-50 
        
            self.x_slope = self.slider_x.getValue()-50
            
            self.n  = self.slider_n.getValue()
            
            self.Frame_number = int(self.slider_frame_number.getValue())
            
            self.number_of_gauss = self.slider_gauss.getValue()
            
            # print slider values
            
            self.text_x.setText("{:.1e}".format(self.x_slope))
            self.text_x.draw()
            
            self.text_y.setText("{:.1e}".format(self.y_slope))
            self.text_y.draw()
            
            self.text_n.setText(int(self.n))
            self.text_n.draw()
            
            self.text_frame_number.setText(self.Frame_number)
            self.text_frame_number.draw()
            
            self.text_gauss.setText(self.number_of_gauss)
            self.text_gauss.draw()
            
            
            # update images
            
            self.update_Frame()
            
            self.update_Frame_corrected () 
            
            self.gauss_smooth ()
            
            self.update_kmeans ()
            
            self.update_display()
        
        pygame.display.quit()
            
    def update_display (self):
        
        self.update_disp_image_data()
        
        self.update_disp_image_label()
        
        SHOW_IMG = np.concatenate([self.disp_data,self.disp_Label],axis = 0) 
    
        self.px = pygame.surfarray.make_surface(SHOW_IMG)
        
        self.screen.blit(self.px, self.px.get_rect())


    def update_Frame (self):
        
        self.Frame = self.Data[:,:,self.Frame_number]
        
    def update_Frame_corrected (self):
        
        x = np.arange(self.Frame.shape[0])  
        y = np.arange(self.Frame.shape[1]) 
        
        xx, yy = np.meshgrid(x,y)
        
        plane = xx * self.x_slope +  yy * self.y_slope 
        
        
        self.Frame_corrected = self.Frame - plane
        
    def update_kmeans (self):
        
        self.centers = self.init_centers()
        
        self.calulate_K_mean()
        
    def gauss_smooth (self):
        
        filtered_image = self.Frame_corrected
        
        for i in range (0,self.number_of_gauss):
        
            kernel = 1/16 * np.array ([[1,2,1],[2,4,2],[1,2,1]])
            
            # filtered_image = cv2.filter2D(img,-1,kernel)
            filtered_image = cv2.blur(filtered_image,(3,3))
        
        self.Frame_corrected = filtered_image
        
        
    def setup(self):

        ## setups images
  
        self.update_disp_image_data()
        self.update_disp_image_label()

        SHOW_IMG = np.concatenate([self.disp_data,self.disp_Label],axis = 0) 
    
        self.px = pygame.surfarray.make_surface(SHOW_IMG)
    
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
        
   
        self.slider_x= pw.Slider(
                self.screen, Window_size [0] - 400 + 25, 110, 150, 10, min = 0, max = 100, step = 1)
        self.slider_x.draw()
        
        self.text_x  = pw.TextBox(
                self.screen, Window_size [0] - 400 + 0, 60, 100, 40, fontSize=24)
        self.text_x.draw()
        
        text_x_describe = pw.TextBox(
                self.screen, Window_size [0] - 400 + 100, 50, 100, 40, fontSize=16,
                borderColour=(128, 128, 128),colour=(128, 128, 128))
        text_x_describe.setText('X tilt')
        text_x_describe.draw()
  
   
        self.slider_y= pw.Slider(
                self.screen, Window_size [0] - 400 + 25, 180, 150, 10, min = 0, max = 100, step = 1) 
        self.slider_y.draw()
        
        self.text_y  = pw.TextBox(
                self.screen, Window_size [0] - 400 + 0, 130, 100, 40, fontSize=24)
        self.text_y.draw()
        
        text_y_describe = pw.TextBox(
                self.screen, Window_size [0] - 400 + 100, 120, 100, 40, fontSize=16,
                borderColour=(128, 128, 128),colour=(128, 128, 128))
        text_y_describe.setText('Y tilt')
        text_y_describe.draw()
        
        
        
        self.slider_n= pw.Slider(
                self.screen, Window_size [0] - 400 + 25, 250, 150, 10, min = 1, max = 10, step = 1,initial = self.n) 
        self.slider_n.draw()
        
        self.text_n  = pw.TextBox(
                self.screen, Window_size [0] - 400 + 0, 200, 60, 40, fontSize=24)
        self.text_n.draw()
        
        text_n_describe = pw.TextBox(
                self.screen, Window_size [0] - 400 + 60, 190, 100, 40, fontSize=16,
                borderColour=(128, 128, 128),colour=(128, 128, 128))
        text_n_describe.setText('Number of clusters')
        text_n_describe.draw()
        
        
        self.slider_gauss= pw.Slider(
                self.screen, Window_size [0] - 400 + 25, 320, 150, 10, min = 0, max = 10, step = 1,initial = 0) 
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
        

    def update_disp_image_data(self):
        im = self.Frame_corrected 
       
        im   = (im - (np.min(im))) 
        im = 255 * (im / np.max(im))
       
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        im = clahe.apply(im.astype(np.uint8))
       
        im = np.transpose(im)

        
        self.disp_data[:, :, 2] = im
        self.disp_data[:, :, 1] = im
        self.disp_data[:, :, 0] = im 
        
    def update_disp_image_label(self):
        im = self.Labels 
       
        im   = (im - (np.min(im))) 
        im = 255 * (im / np.max(im))
       
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        im = clahe.apply(im.astype(np.uint8))
       
        im = np.transpose(im)

        
        self.disp_Label[:, :, 2] = im
        self.disp_Label[:, :, 1] = im
        self.disp_Label[:, :, 0] = im 
        
        
    def Set_done_cmd(self,cmd):
        
        self.done_cmd = cmd


#%% for extraction of data

    def get_all_labels (self):
        
        all_labels = np.empty(self.Data.shape)
        
        for i in range (0,self.number_of_frames):
            self.Frame_number = i
            self.update_Frame()
            self.update_Frame_corrected()
            self.update_kmeans ()
            
            #  add sorting
            # i = np.argsort(self.centers)
            # print(i)
            all_labels [:,:,i] = self.Labels        
        return all_labels
        
    
         
     
        
     
        
        
        



