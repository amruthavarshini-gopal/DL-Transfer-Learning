# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement

Develop an image classification model using transfer learning with the pre-trained VGG19 model.

## DESIGN STEPS
### STEP 1:
Import required libraries and define image transforms.

### STEP 2:
Load training and testing datasets using ImageFolder.

### STEP 3:
Visualize sample images from the dataset.

### STEP 4:
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5:
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6:
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions

## PROGRAM

### Name: Amruthavarshini Gopal
### Register Number: 212223230013

```python
# Load Pretrained Model and Modify for Transfer Learning
model=models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

# Train the model
def train_model(model, train_loader,test_loader,num_epochs=100):
    # Write your code here
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(num_epochs):
      running_loss=0.0
      for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
      train_losses.append(running_loss/len(train_loader))

      model.eval()
      val_loss=0.0
      with torch.no_grad():
        for images,labels in test_loader:
          images,labels=images.to(device),labels.to(device)
          outputs=model(images)
          loss=criterion(outputs,labels.unsqueeze(1).float())
          val_loss+=loss.item();
      val_losses.append(val_loss/len(test_loader))
      model.train()
      print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: Amruthavarshini Gopal")
    print("Register Number: 212223230013")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="622" height="683" alt="Screenshot 2026-05-19 095231" src="https://github.com/user-attachments/assets/3cd6238b-320e-4746-bb6a-1abf858154b7" />

<img width="884" height="664" alt="Screenshot 2026-05-19 095245" src="https://github.com/user-attachments/assets/056044cb-8163-457b-b8cb-4b9c21cbbfa6" />


### Confusion Matrix

<img width="834" height="686" alt="Screenshot 2026-05-19 095302" src="https://github.com/user-attachments/assets/e8a2ac2e-3b2c-4756-911c-8022eb32fa41" />

### Classification Report
<img width="661" height="253" alt="Screenshot 2026-05-19 095311" src="https://github.com/user-attachments/assets/c0f7bf94-fa23-4aff-8b9e-93e411dcf01e" />


### New Sample Data Prediction
<img width="635" height="543" alt="Screenshot 2026-05-19 095329" src="https://github.com/user-attachments/assets/44c95654-f633-4729-8fb3-8232da5c1dc8" />

<img width="604" height="521" alt="Screenshot 2026-05-19 095338" src="https://github.com/user-attachments/assets/d50d6764-37dc-4af9-aa91-bce2b7b83d74" />


## RESULT
VGG19 model was fine-tuned and tested successfully. The model achieved good accuracy with correct predictions on sample test images.
