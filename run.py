from model.teacher_model import *
from model.student_model import *
from utils.data_utils import *
from utils.metrics import *

# Load the data
train_data, test_data = load_CIFAR10()

# Define and Train the teacher ResNet model
teacher_model = TeacherModel()
teacher_model.train_teacher(train_data, learning_rate=0.0001, epochs=2)

# Define and Train the student CNN model
student_model = StudentModel(temperature=3, alpha=0.5, beta=0.5)
student_model.train_student(teacher_model.model, train_data, learning_rate=0.001, epochs=2)

# Evaluate teacher and student models
evaluate_models(teacher_model.model, student_model.model, test_data)