## **Step 4: Partition the Data**

**What are we doing?**  
We are organizing our dataset by dividing it into training, testing, and validation sets. This involves moving images and their corresponding YOLO label files into separate folders.

**Why are we doing it?**  
Partitioning the data ensures the model is trained on one set, validated on another (to tune hyperparameters), and tested on a third (to evaluate performance). This avoids overfitting and provides reliable accuracy results.

**How are we doing it?**

- **Library used**:
    
    - `pathlib`: For clean, readable, and OS-independent file and folder handling.
        
    - `os`: For directory listings and file checks (minimal use).
        
- **Process**:
    
    1. Assume the folders `data/train/images`, `data/test/images`, and `data/val/images` already contain the partitioned image files.
        
    2. Create corresponding `labels` subfolders under each partition (if not already present).
        
    3. For every image in each partition:
        
        - Generate the corresponding `.txt` label filename.
            
        - Check if it exists in `data/labels`.
            
        - If it does, move it into the respective `labels` subfolder.
            

**How does this help the next step?**  
Now each subset (train/test/val) is self-contained with both images and labels. This structure is essential for most object detection pipelines and ensures that models can correctly access matched image-label pairs during training and evaluation.

---

Ready to move on to Step 5?