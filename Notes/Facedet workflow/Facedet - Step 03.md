## **Step 3: Annotate Images**

**What are we doing?**  
We are labeling each image by drawing bounding boxes around the face region using an online tool called [makesense.ai](https://www.makesense.ai/), and exporting those annotations in the YOLO format.

**Why are we doing it?**  
Machine Learning models need **both images and labels** to learn. For object detection, labels are usually bounding boxes around the object of interest. Annotation helps the model learn **where the face is located** in each image.

**How are we doing it?**

- **Tool used**:
    
    - `makesense.ai` – a free web-based annotation tool that allows manual bounding box creation and export in various formats including YOLO.
        
- **Process**:
    
    1. Upload all captured images to makesense.ai.
        
    2. Manually draw bounding boxes around faces in each image.
        
    3. Assign the label (e.g., `"face"`) to each box.
        
    4. Export the annotations in **YOLO format**.
        
- **YOLO Format Overview**:  
    Each image gets a `.txt` file with the **same name** as the image.  
    Each line in the file represents one bounding box in the format:
    
    ```
    <class_id> <x_center> <y_center> <width> <height>
    ```
    
    - All values are **normalized** (range 0 to 1) relative to image width and height.
        
    - For example: `0 0.5 0.5 0.3 0.4` means one object of class 0 at the center of the image.
        

**How does this help the next step?**  
With annotated images and corresponding YOLO labels, we’re now ready to train or test a face detection model using these image-label pairs. The YOLO format is especially useful for object detection frameworks.

---

Let me know when you're ready to move on to Step 4!