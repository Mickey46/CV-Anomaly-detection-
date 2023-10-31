import tkinter as tk
from tkinter import Canvas, NW
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from skimage.transform import resize

# Functions from the provided code
def get_available_camera_indices(max_check=10):
    available_indices = []
    for index in range(max_check):
        cap = cv2.VideoCapture(index)
        ret, _ = cap.read()
        if ret:
            available_indices.append(index)
        cap.release()
    return available_indices

def smooth_filter(img_tensor_batch, target_colors, tolerance):
    distances = torch.norm(img_tensor_batch * 255 - target_colors[:, None, None, :], dim=-1)
    weights = (1 - torch.clamp(distances / tolerance, 0, 1)).unsqueeze(-1)
    return img_tensor_batch * weights

def detect_motion(prev_frame, current_frame, threshold=25):
    """Detect motion by comparing two frames. Returns a binary mask where motion is detected."""
    # Convert frames to uint8 format
    prev_frame_uint8 = (prev_frame * 255).astype(np.uint8)
    current_frame_uint8 = (current_frame * 255).astype(np.uint8)
    
    # Calculate the absolute difference between the two frames
    diff = cv2.absdiff(prev_frame_uint8, current_frame_uint8)
    # Convert the difference to grayscale
    grayscale_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Threshold the grayscale difference to create a binary mask
    _, motion_mask = cv2.threshold(grayscale_diff, threshold, 255, cv2.THRESH_BINARY)
    return motion_mask

# Global variables
camera_indices = get_available_camera_indices()
camera_indices = camera_indices[:5]
caps = [cv2.VideoCapture(index) for index in camera_indices]
color = [(255, 255, 255)]  # Default white color
prev_frames = [None] * len(camera_indices)

# GUI setup
root = tk.Tk()
root.title("Motion Detection")

# Create canvas for image display
canvas = Canvas(root, width=512, height=1024)
canvas.pack()

def update_display():
    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = resize(frame_rgb, (512, 512))
            
            if prev_frames[idx] is not None:
                motion_mask = detect_motion(prev_frames[idx], frame_resized)
                motion_coords = np.argwhere(motion_mask)
                if len(motion_coords) > 0:
                    y, x = motion_coords[np.random.choice(motion_coords.shape[0])]
                    if y < 512:  # Only pick colors from the top half (original image)
                        picked_color = frame_resized[y, x] * 255
                        color[0] = (int(picked_color[0]), int(picked_color[1]), int(picked_color[2]))
            
            image_batch = np.repeat(frame_resized[np.newaxis, ...], 1, axis=0)
            img_tensor_batch = torch.tensor(image_batch)
            target_color_tensor_batch = torch.tensor(np.array([color[0]])).float()
            
            filtered = smooth_filter(img_tensor_batch, target_color_tensor_batch, 100.0).cpu().numpy()[0]
            
            stacked_image = np.vstack([frame_resized, filtered])
            
            # Convert the image to PIL format and then to ImageTk format
            image = Image.fromarray((stacked_image * 255).astype(np.uint8))
            photo = ImageTk.PhotoImage(image=image)
            
            canvas.create_image(0, 0, anchor=NW, image=photo)
            canvas.image = photo
            
            prev_frames[idx] = frame_resized

    # Call update_display() again after 10ms
    root.after(10, update_display)

update_display()

root.mainloop()

