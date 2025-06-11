## **Step 12: Realtime Testing**

---

### **What are we doing?**

We are testing the trained face detection model on a live webcam feed.  
This allows us to see if the model can accurately detect and locate faces in real time.

---

### **Why are we doing it?**

To evaluate how the model performs under real-world, dynamic conditions like changing lighting, angles, and motion.  
It also validates if the model is ready to be deployed in interactive applications like face tracking or AR overlays.

---

### **How are we doing it?**

**Tools/Libraries Used:**

- `cv2.VideoCapture` to access webcam frames.
    
- `TensorFlow` to preprocess and predict using the trained model.
    
- `cv2.rectangle` and `cv2.putText` to draw bounding boxes and labels.
    

**Steps:**

- Start webcam feed using `cv2.VideoCapture(0)`.
    
- For each frame:
    
    - Crop to a 450×450 region for consistent input size.
        
    - Convert BGR to RGB, resize to (120,120), normalize, and expand dims.
        
    - Predict using the model to get class confidence and bounding box.
        
    - If confidence > 0.5, convert the predicted bbox from YOLO to pixel coords.
        
    - Draw the rectangle and label the region as 'face'.
        
- Display the frame and break the loop on 'q' key press.
    

---

### **How does this play into the next step?**

This final step demonstrates a **working face detection system** in action.  
It paves the way for enhancements like **multi-face detection**, **expression tracking**, or integrating with AR devices.

---