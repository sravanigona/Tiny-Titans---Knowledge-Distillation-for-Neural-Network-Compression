import torch
import psutil
import os
import time

# Compute model accuracy
def top_k_accuracy(model, test_data):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in test_data:
            outputs, _ = model(inputs)

            # Top-1 accuracy
            _, top1_preds = torch.topk(outputs, 1, dim=1)
            top1_correct += (top1_preds == labels.view(-1, 1).expand_as(top1_preds)).sum().item()
            
            # Top-5 accuracy
            _, top5_preds = torch.topk(outputs, 5, dim=1)
            top5_correct += (top5_preds == labels.view(-1, 1).expand_as(top5_preds)).sum().item()

            total_samples += labels.size(0)
    
    top1_accuracy = top1_correct / total_samples
    top5_accuracy = top5_correct / total_samples

    return top1_accuracy, top5_accuracy

# Compute model Inference Time and Memory Usage
def compute_efficiency_metrics(model, test_data, num_batches=100):
    model.eval()
    process = psutil.Process(os.getpid())  # Get the current process

    # Measure initial memory usage
    initial_memory = process.memory_info().rss / 1024 ** 2  # Convert to MB

    start_time = time.time()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_data):
            if i >= num_batches:
                break
            _ = model(inputs)  # Forward pass
    end_time = time.time()

    # Measure final memory usage
    final_memory = process.memory_info().rss / 1024 ** 2  # Convert to MB

    # Calculate metrics
    avg_inference_time = (end_time - start_time) / num_batches
    memory_usage = final_memory - initial_memory

    return avg_inference_time, memory_usage

# Compute Compression Ratio
def model_compression_ratio(teacher_model, student_model):
    student_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    teacher_params = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    compression_ratio = teacher_params / student_params

    return student_params, teacher_params, compression_ratio

def evaluate_models(teacher_model, student_model, test_data):
    top1_teacher, top5_teacher = top_k_accuracy(teacher_model, test_data)
    top1_student, top5_student = top_k_accuracy(student_model, test_data)

    print()
    print(f"Teacher Model - Top-1 Accuracy: {top1_teacher:.4f}")
    print(f"Student Model - Top-1 Accuracy: {top1_student:.4f}")
    print(f"Teacher Model - Top-5 Accuracy: {top5_teacher:.4f}")
    print(f"Student Model - Top-5 Accuracy: {top5_student:.4f}")
    print()

    # Compute additional metrics
    student_params, teacher_params, compression_ratio = model_compression_ratio(teacher_model, student_model)
    print("Compression ratio is", compression_ratio)

    # Efficiency Metrics (Inference Time and Memory Usage)
    inference_time_teacher, memory_usage_teacher = compute_efficiency_metrics(teacher_model, test_data)
    inference_time_student, memory_usage_student = compute_efficiency_metrics(student_model, test_data)

    print()
    print(f"Teacher Model - Inference Time: {inference_time_teacher:.4f} seconds")
    print(f"Teacher Model - Memory Usage: {memory_usage_teacher:.2f} MB")
    print()
    print(f"Student Model - Inference Time: {inference_time_student:.4f} seconds")
    print(f"Student Model - Memory Usage: {memory_usage_student:.2f} MB")