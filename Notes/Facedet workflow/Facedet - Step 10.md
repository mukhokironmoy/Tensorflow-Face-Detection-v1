## **Step 10: Train the Model and Plot Performance**

### **Objective**

Train the custom object detection model using the prepared training pipeline and visualize the performance metrics (total loss, classification loss, and regression loss) across epochs.

---

### **1. Define a Custom Training Loop using Keras Subclassing**

We create a class `FaceTracker` that inherits from `tf.keras.models.Model` to customize the training behavior.

#### **Why subclass Model?**

Because the model has **two outputs** (classification and bounding box regression), and we want custom control over how the combined loss is calculated and optimized.

#### **Components:**

- **`__init__()`**: Accepts the compiled face tracker model (backbone).
    
- **`compile()`**: Saves optimizer, classification loss, and localization loss.
    
- **`train_step()`**: Custom logic for:
    
    - Forward pass using the model.
        
    - Compute classification and localization loss.
        
    - Combine losses:  
        `total_loss = localization_loss + 0.5 * classification_loss`
        
    - Backpropagation with gradient tape.
        
- **`test_step()`**: Same logic as train step, but without updating weights.
    
- **`call()`**: Makes the class behave like a callable model (i.e., usable like `model(X)`).
    

---

### **2. Train the Model**

```python
model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss)
```

- We compile the model with:
    
    - `opt`: Adam optimizer with learning rate decay.
        
    - `classloss`: Binary Cross-Entropy for presence of face.
        
    - `regressloss`: Custom localization loss for bounding box.
        

```python
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=40, validation_data=val, callbacks=[tensorboard_callback])
```

- Trains for **40 epochs** using `train` and `val` datasets.
    
- **TensorBoard** logs are also saved for deeper inspection if needed.
    

---

### **3. Plot Training and Validation Performance**

```python
fig, ax = plt.subplots(ncols=3, figsize=(20,5))
```

Plots three graphs to compare performance across epochs:

#### a. **Total Loss**

- `total_loss`: Combination of classification and regression loss.
    
- Helps judge overall learning progress.
    

#### b. **Classification Loss**

- Focuses only on how well the model predicts face presence.
    
- Should steadily decrease if model is learning.
    

#### c. **Regression Loss**

- Evaluates accuracy of predicted bounding box coordinates.
    
- Should also go down over time for better localization.
    

Each plot compares training and validation loss to detect **overfitting** or **underfitting**.

---

### **How does this help next?**

Now that the model is trained and its learning behavior is confirmed through plots, we’re ready to **evaluate it on test images** and visualize its real-world performance.

---

Ready to continue to **Step 11: Evaluate the model on test images and visualize predictions**?