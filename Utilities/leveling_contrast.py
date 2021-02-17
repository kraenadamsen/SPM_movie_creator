"""

Leveling and Contrast

"""


import pygame
import numpy as np 
import pygame_widgets as pw
import cv2




# %%  game methods 


def point_finder(point_str,screen, px):
    image_size = px.get_rect()[2:]
    RUN = True
    while RUN:
    
        pygame.display.update()
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.MOUSEBUTTONUP:
                point_val = event.pos
                                
                if point_val[0] < image_size[0] and point_val[1] < image_size[1]:
                                
                    # print(f'{point_str} point selected')
                    pygame.draw.circle(screen, (0,255,0),point_val,2,4)
                    RUN = False
                    
    return point_val


def Set_done_cmd(cmd):

    global RUN_cmd
    RUN_cmd = cmd
    
def Set_redo_cmd(cmd):
    
    global REDO_cmd
    REDO_cmd = cmd
        
    
def setup(Frame,Refence_dict):        
     
     im = Frame  
     
     im   = (im - (np.min(im))) 
     im = 255 * (im / np.max(im))
     
     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
     im = clahe.apply(im.astype(np.uint8))
     
     
     
     w, h = im.shape
     ret = np.empty((w, h, 3), dtype=np.uint8)
     ret[:, :, 2] = im
     ret[:, :, 1] = im
     ret[:, :, 0] = im 
     
     px = pygame.surfarray.make_surface(ret)
     
     image_size = px.get_rect()[2:]

     Window_size = image_size 
    
     Window_size [0] = Window_size [0] + 200
     Window_size [1] = Window_size [1] + 5
     
     screen = pygame.display.set_mode( Window_size )
     screen.fill((128,128,128))
     screen.blit(px, px.get_rect())
     
     
     
     global button_done     

     button_done = pw.Button(
                screen, Window_size [0] - 200 + 50, 80, 100, 50, text='Done',
                fontSize=22, margin=10,
                inactiveColour=(255, 0, 0),
                pressedColour=(0, 255, 0), radius=20,
                onClick=lambda: Set_done_cmd(False) )
        
     
     global button_redo     

     button_redo = pw.Button(
                screen, Window_size [0] - 200 + 50, 10, 100, 50, text='Redo',
                fontSize=22, margin=10,
                inactiveColour=(255, 0, 0),
                pressedColour=(0, 255, 0), radius=20,
                onClick=lambda: Set_redo_cmd(True)) 
     
     global slider_contrast    

     slider_contrast= pw.Slider(
                screen, Window_size [0] - 200 + 25, 150, 150, 10, min = 0, max = 511,step = 1)
     
     
     
     global text_contrast 
     
     text_contrast = pw.TextBox(
                screen, Window_size [0] - 200 + 25, 170, 60, 40, fontSize=24)
     
     text_contrast_describe = pw.TextBox(
                screen, Window_size [0] - 200 + 85, 170, 100, 40, fontSize=16,
                borderColour=(128, 128, 128),colour=(128, 128, 128))
     text_contrast_describe.setText('Contrast')
     text_contrast_describe.draw()
     
     
     global slider_brightness    

     slider_brightness= pw.Slider(
                screen, Window_size [0] - 200 + 25, 230, 150, 10, min = 0, max = 511, step = 1)
     
     
     global text_brightness
     
     text_brightness = pw.TextBox(
                screen, Window_size [0] - 200 + 25, 250, 60, 40, fontSize=24)
     
     
     text_brightness_describe = pw.TextBox(
                screen, Window_size [0] - 200 + 85, 250, 100, 40, fontSize=16,
                borderColour=(128, 128, 128),colour=(128, 128, 128))
     text_brightness_describe.setText('Brightness')
     text_brightness_describe.draw()
     
     
     global text_frame_number
     
     text_frame_number = pw.TextBox(
                screen, Window_size [0] - 200 + 25, 430, 60, 40, fontSize=24)
     
     
     global slider_frame_number    

     slider_frame_number= pw.Slider(
                screen, Window_size [0] - 200 + 25, 410, 150, 10, min = 1, max = len(Refence_dict), step = 1,initial = 1)
     
     text_frame_describe = pw.TextBox(
                screen, Window_size [0] - 200 + 85, 430, 100, 40, fontSize=16,
                borderColour=(128, 128, 128),colour=(128, 128, 128))
     text_frame_describe.setText('Frame Number')
     text_frame_describe.draw()
     

    
     slider_contrast.draw()     
     slider_brightness.draw()

     
     text_contrast.draw()
     text_brightness.draw()  
    
     button_done.draw()
     
     button_redo.draw()
     
     text_frame_number.draw()
     slider_frame_number.draw()
     
     
     return screen, px

    
def mainLoop(screen, px):
    
    first = second = third = None
         
    first  = point_finder('First point',screen, px)
    second = point_finder('second point',screen, px)
    third  = point_finder('Third point',screen, px)
    
    
    return ( list(first) , list(second) , list(third) )



def update_image (screen_val, Frame, contrast,brightness):
    
    Frame , contrast,brightness = contrast_and_brigthness(Frame, contrast,brightness)
    
    
    
    ret = gray(Frame)
     
    px = pygame.surfarray.make_surface(ret)
    
    
    global screen
    
    screen = screen_val
    
    screen.blit(px, px.get_rect())

    # 
    # Frame  = Frame.transpose()
    return Frame
 

# %% image setup

def truncanate (Frame):
    
    Frame  = np.where(Frame >= 0  , Frame, 0)
    
    Frame  = np.where(Frame <= 255, Frame, 255)
    
    return Frame

def contrast_and_brigthness (Frame, contrast,brightness):
    Frame = (Frame - (np.min(Frame))) 
    Frame = 255 * (Frame / np.max(Frame))   
    
    
    factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
    
    Frame = truncanate(((factor * Frame)   - 128) + 128)
    
    Frame = truncanate(Frame + brightness)
    
    
    return (Frame , contrast, brightness ) 

def gray(im):

     
     w, h = im.shape
     ret = np.empty((w, h, 3), dtype=np.uint8)
     ret[:, :, 2] = im
     ret[:, :, 1] = im
     ret[:, :, 0] = im 
     
     return ret
 
    
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




# %% main loop


def three_point_selection(Data, Refence_dict, Frame_used_for_correction = 1): 
    
    
        ref_file_number,ref_frame_number = Refence_dict [Frame_used_for_correction]
        

        Frame = get_Data(Data,ref_file_number,ref_frame_number) 
        
        # getting real shape 
        w,h = Frame.shape
        
        # resized for display
        Frame_coorected = cv2.resize (Frame,(512,512))

        screen, px = setup(Frame_coorected,Refence_dict)
   
        while RUN_cmd:
            
                global REDO_cmd
                REDO_cmd = False

                
                first , second , third = mainLoop(screen, px)
                
                # rescale selected points 
                scale_para =  512 / w # typically 1 or 2 -  for 512 and 256 image size 
                (first , second , third) = ([int(first[0]  //scale_para),  int(first[1] //scale_para)],
                                            [int(second[0] //scale_para), int(second[1] //scale_para)],
                                            [int(third[0]  //scale_para),  int(third[1] //scale_para)])  
                
                
                first.append (Frame[first[0] ,first[1] ])
                second.append(Frame[second[0],second[1]])
                third.append (Frame[third[0] ,third[1] ])                               
                
                
                
                Frame_coorected = three_point_corrector(first + second + third ,Frame)
                

                while RUN_cmd:    
                    
                    pygame.display.update()
                    events = pygame.event.get()
                    
                    button_done.listen(events)
                    button_done.draw()
                    
                    button_redo.listen(events)
                    button_redo.draw()
                    
                    
                    slider_contrast.listen(events)
                    slider_contrast.draw()
                    
                    slider_brightness.listen(events)
                    slider_brightness.draw()
                    
                    slider_frame_number.listen(events)
                    slider_frame_number.draw()                    
                    
                    text_contrast.setText(slider_contrast.getValue()-255)
                    text_contrast.draw()
                    
                    text_brightness.setText(slider_brightness.getValue()-255)
                    text_brightness.draw()
                    
                    text_frame_number.setText(slider_frame_number.getValue())
                    text_frame_number.draw() 
                    
                    ref_file_number,ref_frame_number = Refence_dict [slider_frame_number.getValue()]
                    
                    Frame = get_Data(Data,ref_file_number,ref_frame_number) 
                    
                    
                    Frame_coorected = three_point_corrector(first + second + third ,Frame)
                    
                    # resized for display
                    Frame_coorected = cv2.resize (Frame_coorected,(512,512))
                    
                    img_corrected = update_image (screen, Frame_coorected, slider_contrast.getValue()-255,slider_brightness.getValue()-255)
                    
                    
                    if REDO_cmd:
                        print('Reselect three points')
                        break 

        
        pygame.display.quit()
        

        
    
        return (first + second + third, img_corrected)


def run (Data, Refence_dict, Frame_used_for_correction = 1):
    
    #%% Command definitions 

    global RUN_cmd
    RUN_cmd = True
    
    global REDO_cmd
    REDO_cmd = False
       

    
    return three_point_selection(Data, Refence_dict, Frame_used_for_correction = 1)
    






