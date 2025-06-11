## **Step 6: Setup the Full Augmentation Pipeline**

**What are we doing?**  
We’re applying our previously tested augmentation pipeline to the **entire dataset**—across `train`, `test`, and `val` partitions—and saving the new images and their updated YOLO labels.

**Why are we doing it?**  
Training a model on just a few raw images may cause overfitting. Augmenting the full dataset increases diversity and quantity, helping the model learn better general features and improving accuracy on unseen data.

**How are we doing it?**  
- **Libraries used**:  
  - `albumentations`: to apply transformations  
  - `cv2`: for reading and saving images  
  - `pathlib`: to handle paths cleanly  
  - Custom `load_yolo_labels()` function: to read labels

- **Process**:  
  1. Iterate through each image in `train`, `test`, and `val` partitions.  
  2. Read the image and its corresponding `.txt` label (if exists).  
  3. If label is missing, create a placeholder box with a tiny bounding box at the corner.  
  4. Apply augmentations **60 times** to each image using `augmentor`.  
  5. Save each augmented image to `aug_data/.../images`.  
  6. Save the corresponding updated label file in YOLO format to `aug_data/.../labels`.  
  7. If no bounding box is returned post-augmentation, write the placeholder box.

**How does this help the next step?**  
Now we have a large and balanced augmented dataset with matching image-label pairs ready for training. The model can now learn from a richer and more varied dataset, leading to better performance.

---

Let me know when you’re ready for **Step 7: Train a Detection Model** or whatever comes next!