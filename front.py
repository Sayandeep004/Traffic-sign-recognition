import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('traffic-sign.keras')

# Class names (Ensure these match the model's output)
class_names ={ 
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)', 
    2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 
    4: 'Speed limit (70km/h)', 
    5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 
    7: 'Speed limit (100km/h)', 
    8: 'Speed limit (120km/h)', 
    9: 'No passing', 
    10: 'No passing veh over 3.5 tons', 
    11: 'Right-of-way at intersection', 
    12: 'Priority road', 
    13: 'Yield', 
    14: 'Stop', 
    15: 'No vehicles', 
    16: 'Veh > 3.5 tons prohibited', 
    17: 'No entry', 
    18: 'General caution', 
    19: 'Dangerous curve left', 
    20: 'Dangerous curve right', 
    21: 'Double curve', 
    22: 'Bumpy road', 
    23: 'Slippery road', 
    24: 'Road narrows on the right', 
    25: 'Road work', 
    26: 'Traffic signals', 
    27: 'Pedestrians', 
    28: 'Children crossing', 
    29: 'Bicycles crossing', 
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing', 
    32: 'End speed + passing limits', 
    33: 'Turn right ahead', 
    34: 'Turn left ahead', 
    35: 'Ahead only', 
    36: 'Go straight or right', 
    37: 'Go straight or left', 
    38: 'Keep right', 
    39: 'Keep left', 
    40: 'Roundabout mandatory', 
    41: 'End of no passing', 
    42: 'End no passing veh > 3.5 tons' 
}

# Function to preprocess the image for the CNN model
def preprocess_image(image, from_video=False):
    if not from_video:
        image = cv2.imread(image)
    image = cv2.resize(image, (32, 32))  # Resize to match the model's input shape
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match batch size
    return image

# Function to predict traffic sign from an image
def predict_image(image_path):
    image = preprocess_image(image_path)

    # Make predictions
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability class

    # Convert numeric prediction to categorical label
    predicted_label = class_names[predicted_class_index]

    # Return the predicted label
    return predicted_label

# Function to upload image and make prediction
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        prediction = predict_image(file_path)
        label.config(text=f"Prediction: {prediction}")
        load_image = Image.open(file_path)
        load_image = load_image.resize((300, 300))
        render = ImageTk.PhotoImage(load_image)
        img_label.config(image=render)
        img_label.image = render

# Function to capture live video and predict traffic signs
def live_video_detection():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame (using the frame directly)
        preprocessed_frame = preprocess_image(frame, from_video=True)
        
        # Predict the traffic sign
        predictions = model.predict(preprocessed_frame)
        predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability class

        # Convert numeric prediction to categorical label
        predicted_label = class_names[predicted_class_index]

        # Display the prediction on the frame
        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Video Traffic Sign Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Create a Tkinter window
window = tk.Tk()
window.title("Traffic Sign Recognition")

# Create and place widgets
label = Label(window, text="Upload an image or start live video")
label.pack()

upload_btn = Button(window, text="Upload Image", command=upload_image)
upload_btn.pack()

img_label = Label(window)
img_label.pack()

video_btn = Button(window, text="Start Live Video", command=live_video_detection)
video_btn.pack()

# Run the GUI loop
window.mainloop()
