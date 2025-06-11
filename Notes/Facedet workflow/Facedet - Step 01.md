## **Step 1: Collect Images**

**What are we doing?**  
We are capturing 30 images from the webcam and saving them locally with unique names. This creates a dataset of real-time face images.

**Why are we doing it?**  
A model needs input data to learn or predict. Collecting our own images ensures the data reflects our use case (lighting, angle, camera, etc.).

**How are we doing it?**

- **Libraries used**:
    
    - `cv2` to access the webcam and handle images
        
    - `pathlib` to manage file paths
        
    - `uuid` to assign unique names
        
    - `time` to space out captures
        
- **Process**:
    
    1. Create a directory for images (`data/images`).
        
    2. Start webcam (`cv2.VideoCapture(0)`).
        
    3. Loop 30 times: read frame → save it with a unique name → display it briefly.
        
    4. Exit early if 'q' is pressed.
        
    5. Release webcam and close windows.
        

**How does this help the next step?**  
We now have a folder of captured images which can be preprocessed and fed into a face detection model.

---

