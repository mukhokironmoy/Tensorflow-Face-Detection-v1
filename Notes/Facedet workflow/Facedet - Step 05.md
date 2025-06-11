## **Step 5: Setup Primary Image Augmentation (Test on One Image)**

**What are we doing?**  
We are testing a set of image augmentations on a single image and its label to ensure that transformations apply correctly to both the image and its bounding box.

**Why are we doing it?**  
Image augmentation improves model generalization by simulating variations (like brightness, flips, crops). Testing on one image helps verify that bounding boxes still match the object after transformation.

**How are we doing it?**

- **Libraries used**:
    
    - `albumentations`: to define and apply multiple augmentations easily
        
    - `cv2`: to read and draw on images
        
    - `matplotlib`: to display images
        
    - `Pathlib`: to handle file paths
        
- **Process**:
    
    1. Define an `albumentations.Compose` pipeline with transformations like crop, flip, brightness, etc.
        
    2. Read one image and its YOLO `.txt` label file.
        
    3. Parse YOLO format: class + normalized `x_center y_center width height`.
        
    4. Apply augmentations using the parsed bounding box and label.
        
    5. Convert YOLO bbox to pixel corners using image shape.
        
    6. Draw the new bounding box on the augmented image.
        
    7. Display the result to visually confirm correctness.
        

**How does this help the next step?**  
Once validated, this augmentation pipeline can be applied to the entire dataset to enrich training data while keeping labels accurate — boosting model robustness.

---

Let me know when you're ready for Step 6!