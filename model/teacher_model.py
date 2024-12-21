import torch.nn as nn
import torch
from torchvision.models import resnet18, ResNet18_Weights


class ResNetWithFeatures(nn.Module):
    def __init__(self, base_model):
        super(ResNetWithFeatures, self).__init__()
        # Use ResNet backbone
        self.base_model = nn.Sequential(*list(base_model.children())[:-2])  # Remove last FC layer
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, 10)

    def forward(self, x):
        features = []
        for name, layer in self.base_model.named_children():
            x = layer(x)
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:  # Extract features from specific layers
                features.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits, features

class TeacherModel:
    def __init__(self):
        weights = ResNet18_Weights.IMAGENET1K_V1  # Use the ImageNet weights
        base_model = resnet18(weights=weights)  # Load the model with pretrained weights
        self.model = ResNetWithFeatures(base_model)

        # Adjust output layer for CIFAR-10 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, 10) 

    def train_teacher(self, train_data, learning_rate=0.0001, epochs=10):
        # Fine-tuning the teacher model
        for param in self.model.parameters():
            param.requires_grad = True  # Ensure the model is trainable

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  # Small learning rate for fine-tuning

        # Training the teacher model
        print("Training the teacher model...")
        self.model.train()  # Set the model to training mode

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_data:
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs, _ = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_data):.4f}")
