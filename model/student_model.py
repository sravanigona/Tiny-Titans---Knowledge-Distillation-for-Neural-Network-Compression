import torch.nn as nn
import torch
import torch.nn.functional as F


# Student: Custom CNN with matching output dimensions
class StudentCNN(nn.Module):
    def __init__(self):
        super(StudentCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Added an extra layer for feature matching
        self.fc2 = nn.Linear(256, 10)  # Ensure output layer matches the number of classes

    def forward(self, x):
        features = []
        x = F.relu(self.conv1(x))
        features.append(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        features.append(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        features.append(x)
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits, features
    
class StudentModel:
    def __init__(self, temperature=3, alpha=0.5, beta=0.5):
        self.model = StudentCNN()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    # Knowledge Distillation Loss Function
    def distillation_loss(self, student_outputs, teacher_outputs, labels, student_features, teacher_features):
        # Logit Matching Loss (Soft Target)
        logit_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_outputs / self.temperature, dim=1),
                                                        F.softmax(teacher_outputs / self.temperature, dim=1)) * (self.temperature ** 2)
        
        # Feature Matching Loss (Soft Target)
        mse_loss = nn.MSELoss()
        feature_loss = sum(mse_loss(sf, tf) for sf, tf in zip(student_features, teacher_features)) * self.beta
        
        # Hard Target Loss (Cross-entropy)
        hard_loss = F.cross_entropy(student_outputs, labels)

        # Combined Loss
        return self.alpha * (logit_loss + feature_loss) + (1 - self.alpha) * hard_loss

    # Training Function for the Student Model
    def train_student(self, teacher, train_data, learning_rate=0.0001, epochs=10):
        # Freeze the teacher model
        teacher.eval()  

        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        print("Training the student model...")
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_data:
                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_outputs, teacher_features = teacher(inputs)
                
                student_outputs, student_features = self.model(inputs)
                loss = self.distillation_loss(student_outputs, teacher_outputs, labels, student_features, teacher_features)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_data)}")