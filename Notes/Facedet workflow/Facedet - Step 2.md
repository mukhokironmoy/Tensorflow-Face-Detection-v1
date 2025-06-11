## **Step 2: Load and Visualize the Images**

**What are we doing?**  
We are loading the images we collected in Step 1 and displaying them in a grid format to verify that they’ve been captured correctly.

**Why are we doing it?**  
Before we move to preprocessing or modeling, we need to confirm that the images are readable and correctly formatted. Visualization helps spot issues like corrupted files or incorrect captures.

**How are we doing it?**

- **Libraries used**:
    
    - `tensorflow` to efficiently load and process image files
        
    - `matplotlib` to visualize images
        
    - `cv2`, `json`, `numpy` are imported (though not directly used in this step)
        
- **Process**:
    
    1. List all `.jpg` files from `data/images` using `tf.data.Dataset.list_files`.
        
    2. Define a `load_image` function to read and decode each image.
        
    3. Use `.map()` to apply this function to all images.
        
    4. Group images into batches of 4 for easier plotting.
        
    5. Fetch the first batch and plot the images in a 1-row grid.
        

**How does this help the next step?**  
Now that we’ve verified our dataset is correctly loaded and formatted, we can safely proceed to tasks like preprocessing, face detection, or training a model.

---

Let me know when you're ready for Step 3!