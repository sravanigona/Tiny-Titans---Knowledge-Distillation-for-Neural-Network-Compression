# Knowledge Distillation for Model Compression

This project implements a **Knowledge Distillation (KD)** framework to compress a large, complex neural network (ResNet-18 teacher model) into a lightweight, efficient custom CNN student model. The focus is on optimizing the model for deployment in resource-constrained environments without significantly compromising accuracy. ([Full Project Report](https://github.com/sravanigona/Tiny-Titans---Knowledge-Distillation-for-Neural-Network-Compression/blob/main/Project%20Report.pdf))

## Objective

- To reduce model size and inference time while maintaining high accuracy using KD techniques.
- To enable efficient deployment of deep learning models on devices with limited computational resources, such as IoT devices or mobile platforms.

## Features

- **Logit Matching:** Transfers knowledge by mimicking the softened output probabilities of the teacher model.
- **Feature Matching:** Enhances learning by aligning intermediate feature representations between the teacher and student models.
- **Scalability:** Tested on both CIFAR-10 and ImageNet datasets to ensure generalizability across different dataset scales.

## Methodology

1. **Teacher Model:** ResNet-18 trained on CIFAR-10 and ImageNet datasets.
2. **Student Model:** Custom lightweight CNN designed for efficiency.
3. **Knowledge Transfer Techniques:**
   - **Logit Matching:** Used to capture nuanced class relationships.
   - **Feature Matching:** Ensured the student model inherits hierarchical feature representations.
4. **Evaluation Metrics:** Model accuracy, inference time, and memory usage were measured for both teacher and student models.


## Results

- **Performance Efficiency:** Achieved 40% faster inference time compared to the teacher model.
- **Memory Optimization:** Reduced memory usage by a factor of 5x.
- **Accuracy Preservation:** Maintained high accuracy with only a 2-4% performance gap compared to the teacher model.


## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/sravanigona/Tiny-Titans---Knowledge-Distillation-for-Neural-Network-Compression.git
2. Install pipenv and start virtual env
   ```
   pip install pipenv
   pipenv shell
3. Install required packages
   ```
   pipenv install
4. Train and evaluate the student model
   ```
   python run.py
