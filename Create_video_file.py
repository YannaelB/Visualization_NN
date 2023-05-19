import os
import imageio
import cv2


    # Name of the folder with the images with which you want to create a video
folder_name = "pict_model_AI_3"

    # Get a list with the name of each image/file in the folder
image_files = sorted([os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.endswith('.png')])


    # If you want to create a .gif file 
with imageio.get_writer('mygiftest2.7.gif', mode='I') as writer:
    for filename in image_files:
        image = imageio.imread(filename)
        writer.append_data(image)



    # Name of the folder with the images with which you want to create a video
folder_name = "pict_model_AI_3"


# Get the size of one image to define the size of the video
first_image = cv2.imread(image_files[0])
height, width, _ = first_image.shape
size = (width, height)


# Create the video
video_writer = cv2.VideoWriter('myvideo3.1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, size)
for i in range(1,len(image_files)+1):
    filename = f"pict_model_AI_3\model_Ai2.1{i}.png"
    image = cv2.imread(filename)
    video_writer.write(image)