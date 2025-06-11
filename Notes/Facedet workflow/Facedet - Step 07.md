## **Step 7: Load Images and Labels into TensorFlow and Visualise**

**What are we doing?**  
We’re loading the augmented dataset into TensorFlow, preparing it for training, and visually verifying that images and labels are correctly paired and processed.

**Why are we doing it?**  
TensorFlow models require image and label data in a structured pipeline. Visualizing them helps confirm that bounding boxes align properly with the objects after all preprocessing steps.

**How are we doing it?**

- **Libraries used**:
    
    - `TensorFlow`: for dataset loading, preprocessing, batching
        
    - `cv2`, `matplotlib`: for visualization
        
    - `numpy`: for bounding box handling
        
- **Process**:
    
    1. Use `tf.data.Dataset.list_files()` to load all images and labels from `aug_data`.
        
    2. Resize each image to **120x120** and normalize pixel values between 0–1.
        
    3. Use a custom function (`new_load_yolo_labels`) with `tf.py_function()` to read YOLO label `.txt` files.
        
    4. Pair images with their corresponding labels using `tf.data.Dataset.zip()`.
        
    5. Shuffle, batch (size 8), and prefetch data to optimize training.
        
    6. Visualize sample images with bounding boxes by converting YOLO coordinates to pixel corner format and drawing rectangles.
        

**How does this help the next step?**  
We now have a properly formatted and verified TensorFlow dataset ready to be passed directly into a model training loop, ensuring our data is accurate and efficiently structured.

---

Ready for **Step 8: Build and Train the Model** whenever you are!