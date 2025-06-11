## **Step 8: Build the Model and Do a Trial Run**

**What are we doing?**  
We’re building a dual-output deep learning model using a pre-trained backbone and testing it on a sample batch to get initial predictions.

**Why are we doing it?**  
This model will simultaneously classify if a face exists in an image and predict the bounding box around it — a foundational setup for face detection systems.

**How are we doing it?**

- **Libraries used**:
    
    - `tensorflow.keras.models`, `layers`, `applications`: for model building
        
    - `VGG16`: used as a pre-trained feature extractor (without top classification layers)
        
- **Process**:
    
    1. Define an `Input` layer of shape (120, 120, 3).
        
    2. Pass it through the **VGG16** model (excluding top FC layers) to extract features.
        
    3. Split into two heads:
        
        - **Classification head**: uses `GlobalMaxPooling2D`, followed by two Dense layers; outputs a **sigmoid** value (face present or not).
            
        - **Regression head**: uses the same pooled features, predicts **4 normalized coordinates** for bounding boxes.
            
    4. Combine both outputs into a single Keras `Model`.
        
    5. Run a quick trial by passing a batch of images to check predicted class probabilities and coordinates.
        

**How does this help the next step?**  
This base model is now ready for **compilation and training**. The trial run confirms the model outputs are in expected format, allowing us to proceed to actual learning and evaluation.

---

Let me know when you're ready to continue with Step 9: Compile and Train the Model!