## **Step 11: Make Predictions with the Trained Model**

---

### **What are we doing?**

We are using our trained face detection model to make predictions on unseen test data.  
The goal is to check whether the model correctly detects faces and draws bounding boxes around them.

---

### **Why are we doing it?**

This helps us visually verify how well our model generalizes to new data.  
It's a practical way to assess real-world performance before deployment.

---

### **How are we doing it?**

**Tools/Libraries Used:**

- `TensorFlow` for loading test data and running predictions.
    
- `cv2 (OpenCV)` for drawing bounding boxes on images.
    
- `matplotlib` for visualizing the results.
    

**Steps:**

- First, get a batch of test images using `.as_numpy_iterator().next()`.
    
- Then, use `facetracker.predict(...)` to get predicted class probabilities and bounding boxes.
    
- A helper function converts YOLO-style normalized bounding boxes into pixel corner coordinates.
    
- For each image:
    
    - If the predicted class probability > 0.9, draw a rectangle using OpenCV.
        
    - Display the image using `matplotlib`, without axis labels.
        

---

### **How does this play into the next step?**

We now have visual proof of the model's predictions, showing where it detects faces.  
This sets the stage for integrating the model into a **real-time system** using webcam feeds for live inference and tracking.

---