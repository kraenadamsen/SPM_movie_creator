"""
Selecting frames for movie 

"""
import pygame
import pygame_widgets as pw
import cv2
import numpy as np


# %%

def Set_done_cmd(cmd):
    
    global RUN_cmd
    RUN_cmd = cmd

def gray(im,im_size):
    
    
    im = cv2.resize(im, dsize=(im_size, im_size), interpolation=cv2.INTER_CUBIC)
     
    im = (im - (np.min(im))) 
    im = 255 * (im / np.max(im))
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im = clahe.apply(im.astype(np.uint8))
    
    
    ret = np.repeat(im[:, :, np.newaxis], 3, axis=2)
    
    
    return ret


def get_number_of_images(file):
    
    num = 0
    
    for entry in file:
        im_num = ''.join(filter(str.isdigit, entry))
        
        if not im_num:
            im_num = 0 
        
        im_num = int(im_num)
        
        if im_num > num:
            num = im_num
     
    return num



def setup ():
    

    # %% crete pygame window
    Window_size = [0,0] 
    
    Window_size [0] = 256*4
    Window_size [1] = 256+80 
    
    screen = pygame.display.set_mode( Window_size )
    screen.fill((128,128,128))
    
    update_view (screen,1,sum(number_of_im))
    
    #%% input widgets 
    
    global slider_first_frame    

    slider_first_frame = pw.Slider(
                screen, 25, 256 + 20, 200, 10, min = 1, max = sum(number_of_im),step = 1,initial = 1)
    
    global text_first 
     
    text_first = pw.TextBox(
                screen ,25, 256 + 35, 60, 40, fontSize=24)
    
    text_first.draw()    
    slider_first_frame.draw()

    


    global slider_last_frame    

    slider_last_frame = pw.Slider(
                screen, 1024-256+25, 256 + 20, 200, 10, min = 1, max = sum(number_of_im),step = 1,initial = sum(number_of_im))
    
    global text_last 
     
    text_last = pw.TextBox(
                screen ,1024-256+25, 256 + 35, 60, 40, fontSize=24)
    
    text_last.draw()
    slider_last_frame.draw()

    global button_done    

    button_done = pw.Button(
               screen, 512 - 25, 256 +20, 50, 40, text='done',
               fontSize=22, margin=10,
               inactiveColour=(128, 200, 128),
               pressedColour=(0, 255, 0), radius=20,
               onClick=lambda: Set_done_cmd(False))     


    pygame.display.update() 
    
    
    return screen 


def get_frame (Frame_num):
    
    ref_file_number, ref_frame_number =  refence_dict [Frame_num]   
    
    
    VZ_scale =  Data[ref_file_number][f'IMG_{ref_frame_number:03d}']['Feedback'][0,0][0][0][-1][0]
    AAperVZ  =  Data[ref_file_number][f'IMG_{ref_frame_number:03d}']['Config'][0,0][0][0][-1][0]
    img_data =  Data[ref_file_number][f'IMG_{ref_frame_number:03d}']['Data'][0,0] / 32768 * VZ_scale * AAperVZ
    
    Frame = (img_data - np.mean(img_data))*10**-10 # units in meters


    Frame  = Frame [::-1,:]
    Frame = Frame.transpose()
    
    return Frame

def update_view (screen,first_frame_num,last_frame_num):

    First_Frame = get_frame(first_frame_num)    
    Last_Frame  = get_frame(last_frame_num)
    
    First_Frame = gray(First_Frame,256)
    Last_Frame  = gray(Last_Frame ,256)
     
    px_first_frame = pygame.surfarray.make_surface(First_Frame)
    px_last_frame  = pygame.surfarray.make_surface(Last_Frame )
    
    
    # # %% creating the movie strip 
    # frame_increament = (last_frame_num-first_frame_num)//3
    
    # Frame_1 = gray( get_frame(frame_increament*1+first_frame_num),150)
    # Frame_2 = gray( get_frame(frame_increament*2+first_frame_num),150)
    # Frame_3 = gray( get_frame(frame_increament*3+first_frame_num),150)
    
    # px_strip_1 = pygame.surfarray.make_surface(Frame_1)
    # px_strip_2 = pygame.surfarray.make_surface(Frame_2)
    # px_strip_3 = pygame.surfarray.make_surface(Frame_3)
    
    # # %% "plotting" images

    # screen.blit(px_strip_1, (256 + 1*15, 256-150 ,150,150))
    # screen.blit(px_strip_2, (256 + 2*15 + 150, 256-150 ,150,150))
    # screen.blit(px_strip_3, (256 + 3*15 + 300, 256-150 ,150,150))
    

    
    screen.blit(px_first_frame, (0,0,256,256))
    screen.blit(px_last_frame , (3*256,0,256,256))
    
    pygame.display.update()


    


# %% main loop

def main_loop(number_of_files,number_of_im):
    
    pygame.init()
    screen = setup ()
    
    
    
    while RUN_cmd:
        
        
        
        pygame.display.update()
        events = pygame.event.get()

        slider_first_frame.listen(events)
        slider_first_frame.draw()
        
        slider_last_frame.listen(events)
        slider_last_frame.draw()
        

        
        
        text_first.setText(slider_first_frame.getValue())
        text_first.draw()
        
        
        text_last.setText(slider_last_frame.getValue())
        text_last.draw()
        
        
        
        
        update_view (screen,slider_first_frame.getValue(),slider_last_frame.getValue())
    
        button_done.listen(events)
        button_done.draw()
        
    pygame.display.quit()
        
    
    return (slider_first_frame.getValue(),slider_last_frame.getValue())
# %%


def run (Data_raw):
    
    global RUN_cmd 
    RUN_cmd = True
    
    global Data
    
    Data = Data_raw
    
    global number_of_files 
    
    number_of_files = len(Data_raw)
    
    global number_of_im 
    
    number_of_im = []
    
    for file in Data_raw:
        number_of_im.append( get_number_of_images(file) )
        
    
    # %% make dictionary of frames vs. file and image number
    
    file_ref = []
    
    n = 0
    
    for num_in_file in number_of_im:
        list_ =  num_in_file * [n]
        image_num = range(1,num_in_file+1)
        
        comb = list(zip(list_,image_num))
       
        file_ref = file_ref + comb
        
        n += 1        
        
    frame_ref =list ( range(1,sum(number_of_im)+1))
    
    global refence_dict
    
    refence_dict = dict(zip(frame_ref,file_ref))
    
    # inv_refence_dict = {v: k for k, v in refence_dict.items()}
        
    # %%

    (first_frame,last_frame) = main_loop (number_of_files,number_of_im)

   
    frame_list  = list(range(first_frame,last_frame))
    
    used_frames = list( map(refence_dict.get, frame_list) )
    
    
    refence_dict = dict(zip(list(range(1,len(used_frames))),used_frames))
    
    
    return refence_dict
