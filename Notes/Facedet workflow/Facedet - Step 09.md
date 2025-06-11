## **Step 9: Define Losses and Optimizers and Test the Functions**

**What are we doing?**  
We're defining the optimizer and loss functions required to train our dual-output face detection model, and testing if they compute correctly on a sample batch.

**Why are we doing it?**  
Loss functions guide the model on how to improve predictions. The optimizer updates model weights based on these losses. Testing them ensures they’re working as expected before training.

**How are we doing it?**

- **Libraries/Tools used**:
    
    - `tf.keras.optimizers.Adam`: for adaptive gradient-based optimization.
        
    - `tf.keras.losses.BinaryCrossentropy`: for classification loss (face or no face).
        
    - Custom `localization_loss` function: to penalize incorrect bounding boxes.
        
- **Process**:
    
    1. Compute `batches_per_epoc` to dynamically adjust learning rate decay.
        
    2. Initialize **Adam optimizer** with a base learning rate and decay.
        
    3. Define:
        
        - `classloss`: standard binary cross-entropy for classification.
            
        - `regressloss`: custom function comparing predicted vs. true bounding box center and size.
            
    4. Test these losses using the predicted outputs from Step 8.
        

**How does this help the next step?**  
With losses and optimizer working correctly, we're ready to **compile and train** the model — which is the core of model learning and improvement.

---

Ready for **Step 10: Compile and Train the Model**?