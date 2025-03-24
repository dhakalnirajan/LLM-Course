# Large Language Model (LLM) Course

A comprehensive course on Large Language Models, from the fundamentals to advanced techniques and applications.

## Overview

This course provides a deep dive into the world of Large Language Models (LLMs). We'll cover the theoretical foundations, practical implementation, and cutting-edge research in this rapidly evolving field. The course is designed for students with a background in [Prerequisites - e.g., machine learning, Python programming, linear algebra].  By the end of this course, you will be able to:

* Understand the core concepts behind LLMs (e.g., language modeling, transformers, attention).
* Implement and train your own LLMs from scratch.
* Fine-tune pre-trained models for specific tasks.
* Explore advanced topics like inference optimization, reinforcement learning, and multimodal models.
* Deploy LLM-powered applications.

## Syllabus

<!--  This section mirrors the directory structure and provides links to each chapter's README. -->

The course is structured into the following chapters:

1. [**Chapter 01: Bigram Language Model**](01_Bigram_Language_Model/README.md) - Introduction to language modeling and the bigram model.
2. [**Chapter 02: Micrograd**](02_Micrograd/README.md) - Building a simple automatic differentiation engine.
3. [**Chapter 03: N-gram Model**](03_N-gram_Model/README.md) - Extending the bigram model to n-grams.
4. [**Chapter 04: Attention**](04_Attention/README.md) - Understanding the attention mechanism.
5. [**Chapter 05: Transformer**](05_Transformer/README.md) - Implementing the Transformer architecture.
6. [**Chapter 06: Tokenization**](06_Tokenization/README.md) - Exploring different tokenization techniques (minBPE, byte pair encoding).
7. [**Chapter 07: Optimization**](07_Optimization/README.md) - Optimization algorithms for LLMs (AdamW).
8. [**Chapter 08: Need for Speed I: Device**](08_Need_for_Speed_I_Device/README.md) - Optimizing for different devices (CPU, GPU).
9. [**Chapter 09: Need for Speed II: Precision**](09_Need_for_Speed_II_Precision/README.md) - Mixed precision training (fp16, bf16, fp8).
10. [**Chapter 10: Need for Speed III: Distributed**](10_Need_for_Speed_III_Distributed/README.md) - Distributed training (DDP, ZeRO).
11. [**Chapter 11: Datasets**](11_Datasets/README.md) - Working with datasets for LLMs.
12. [**Chapter 12: Inference I: kv-cache**](12_Inference_I_kv-cache/README.md) - Optimizing inference with kv-caching.
13. [**Chapter 13: Inference II: Quantization**](13_Inference_II_Quantization/README.md) - Quantization techniques for faster inference.
14. [**Chapter 14: Finetuning I: SFT**](14_Finetuning_I_SFT/README.md) - Supervised fine-tuning (SFT).
15. [**Chapter 15: Finetuning II: RL**](15_Finetuning_II_RL/README.md) - Reinforcement learning for fine-tuning (RLHF, PPO, DPO).
16. [**Chapter 16: Deployment**](16_Deployment/README.md) - Deploying LLM-powered applications (API, web app).
17. [**Chapter 17: Multimodal**](17_Multimodal/README.md) - Exploring multimodal models (VQVAE, diffusion transformer).

**Appendix:**

* [**Programming Languages**](Appendix/Programming_Languages/README.md) - Overview of relevant programming languages (Assembly, C, Python).
* [**Data Types**](Appendix/Data_Types/README.md) -  Integer, Float, String, and various encodings.
* [**Tensor Shapes**](Appendix/Tensor_Shapes/README.md) - Understanding tensor operations.
* [**Deep Learning Frameworks**](Appendix/Deep_Learning_Frameworks/README.md) - PyTorch and JAX
* [**Neural Net Architecture**](Appendix/Neural_Net_Architecture/README.md) - Popular Architectures.
* [**Multimodal**](Appendix/Multimodal/README.md) - Discussion on other multimodal models.

## Prerequisites

* **Programming:**  Strong proficiency in Python.  Familiarity with C/C++ is a plus, but not strictly required.
* **Machine Learning:**  Solid understanding of machine learning fundamentals (e.g., supervised learning, neural networks, loss functions, optimization).
* **Linear Algebra:**  Comfortable with basic linear algebra concepts (vectors, matrices, dot products, matrix multiplication).
* **Calculus:**  Basic understanding of derivatives and gradients.
* **Deep Learning Frameworks:**  Prior experience with PyTorch is highly recommended. Experience with JAX is beneficial, but will be covered in the Appendix.

## Setup

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/dhakalnirajan/LLM-Course.git
    cd LLM_Course
    ```

2. **Create a Virtual Environment (Recommended):**

    ```bash
    python3 -m venv venv  # Or use conda: conda create -n llm_course python=3.9
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Note:* The `requirements.txt` file will be populated as you add dependencies throughout the course.

4. **Jupyter Notebook (Optional but Recommended):**

    ```bash
    pip install jupyterlab  # Or jupyter notebook
    jupyter lab  # Or jupyter notebook
    ```

## Contributing

Contributions are welcome!  If you find any errors, have suggestions for improvements, or want to add new content, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix: `git checkout -b my-feature`.
3. Make your changes and commit them with clear, descriptive messages.
4. Push your branch to your forked repository: `git push origin my-feature`.
5. Submit a pull request to the `main` branch of this repository.

## License

<!--  State the license under which the project is released. -->
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
