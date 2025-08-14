# 🌀 Riemannian Flow Matching (RFM) on General Geometries

We study Riemannian Flow Matching on Stacked-MNIST, CIFAR-10, Tiny-ImageNet-200 using hyperspherical, hyperbolic, and toroidal latent manifolds, each equipped with an autoencoder and classifier.

> **Generative Modeling on High-Dimensional Non-Euclidean Manifolds** — a unified Flow Matching framework enabling efficient generation and reconstruction on complex geometries such as spheres, hyperbolic spaces, and tori.

---

## 📖 Detailed Task Description

### **Task Pipeline Overview**
![Task Pipeline Overview](images/task_pipeline.jpg)

---

### **Data Construction – Stacked-MNIST**
- Randomly select three digits and assign each to an RGB channel.

![Stacked-MNIST RGB Construction](images/stacked_mnist_rgb.jpg)

---

### **Manifold Mapper Architecture**
#### Stacked-MNIST
![Encoder-Decoder Architecture](images/encoder_decoder.png)

#### CIFAR-10
![Encoder-Decoder Architecture](images/ae.png)

#### Tiny-ImageNet-200
![Encoder-Decoder Architecture](images/ae1.png)

---

### **Generation Results on Different Manifolds**
#### Hypersphere
![Hypersphere Generation](images/1.png)

#### Hyperbolic
![Hyperbolic Generation](images/2.png)

#### FlatTorus (UMAP 2-D Visualization)
![FlatTorus Generation](images/3.png)

---

### **CIFAR-10 Generation & Reconstruction on Hypersphere**
- **Reconstruction**
![CIFAR-10 Hypersphere](images/cifar10_hypersphere_recon.png)

- **Random Generation**
![CIFAR-10 Hypersphere](images/cifar10_hypersphere_gen.png)

---

### **Tiny-ImageNet-200 Experimental Results**
- **Reconstruction (128-D Latent)**
![Tiny-ImageNet Recon](images/tiny_imagenet_recon.png)

- **Reconstruction (256-D Latent)**
![Tiny-ImageNet Recon](images/256.png)

- **Random Generation**
![Tiny-ImageNet Gen](images/tiny_imagenet_gen.png)

---

## 📂 File Structure
```bash
├── Datasets-Stacked_MNIST/        # Datasets (Stacked-MNIST; CIFAR-10 and Tiny-ImageNet-200 not included)
├── Auto-Encoder/                  # Encoder / Decoder
├── Classifier/                    # Classifiers
├── Riemannian-Flow-Matching/      # RFM code
├── images/                        # README figures
└── README.md
```