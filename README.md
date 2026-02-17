# CelebA-Face-Enhancement-MLProject

**Advanced Face Image Restoration & Super-Resolution using Deep Learning.**

This repository contains the implementation of a robust Machine Learning pipeline designed to restore high-quality face images from degraded inputs.
---

##  Key Features & Highlights

This project goes beyond basic denoising by implementing several advanced features:

*   **Composite Loss Function:** The model is trained using a weighted combination of:
    *   **Restoration Loss (L1/L2):** To minimize pixel-level differences.
    *   **Identity/Perceptual Loss:** Using a pre-trained face recognition model (e.g., AdaFace/ArcFace) to ensure the restored face matches the identity of the original subject (Embedding Distance minimization).
*   **Complex Degradation Simulation:** The model learns to handle multiple types of image corruption:
    *    Low Resolution (Downsampling)
    *    Gaussian Noise
    *    **Motion Blur** (Advanced/Bonus Feature)
*   **Face Alignment:** Preprocessing pipeline includes face detection and landmark alignment (eyes, nose, mouth) for consistent model input.
*   **Generalization Test:** Evaluated the model's performance on an entirely different dataset (unseen during training) to prove robustness.

##  Project Structure

*   `p1 (1).ipynb`: The core Jupyter Notebook containing:
    *   Data loading and augmentation pipeline.
    *   Custom Dataset class for on-the-fly degradation.
    *   Model Architecture (Encoder-Decoder/U-Net).
    *   Training loop with Composite Loss.
    *   Evaluation metrics and visualization.
*   `MLP1-report.pdf`: Comprehensive technical report detailing the mathematical background, architecture decisions, loss function analysis, and final results.
*   `Project1.pdf`: Original problem statement and requirements.

##  Methodology

### 1. Data Pipeline & Preprocessing
We utilized the **CelebA** dataset.
*   **Face Alignment:** Images are cropped and aligned based on facial landmarks.
*   **Degradation:** High-quality images serve as ground truth (targets). Inputs are generated dynamically by applying:
    *   Resizing (Downscaling).
    *   Adding Gaussian noise (variable intensity).
    *   Applying Motion Blur kernels.

### 2. Model Architecture
A **Convolutional Neural Network (CNN)** based on an Encoder-Decoder design (similar to U-Net) is used to map degraded images to high-resolution outputs. The network learns to capture spatial hierarchies and reconstruct fine facial details.

### 3. Training Strategy
To achieve photorealistic results while maintaining identity, the total loss function is defined as:
$$ L_{total} = \lambda_{rec} L_{reconstruction} + \lambda_{id} L_{identity} $$
Where $L_{identity}$ is calculated as the distance between the embeddings of the generated face and the ground truth face extracted by a recognition network.

##  Evaluation & Results

The model performance is evaluated using both quantitative metrics and qualitative visual inspection.

### Quantitative Metrics
*   **PSNR (Peak Signal-to-Noise Ratio):** Measures reconstruction quality.
*   **SSIM (Structural Similarity Index):** Measures structural similarity.
*   **Embedding Distance:** Measures how well the identity is preserved.

*(For detailed charts, loss curves, and specific numbers, please refer to `MLP1-report.pdf`)*

### Qualitative Results
The model successfully removes noise and deblurs images. Below is an example of the pipeline:
*   **Input:** Blurry, Noisy, Low-Res Image.
*   **Output:** Sharp, Clean, High-Res Image with preserved identity.

##  Limitations & Future Improvements

Due to **hardware constraints (GPU limitations)** and time restrictions, the model presented in this repository was trained for only **2 epochs**.

*   **Training Duration:** While the model shows promising results, the loss curves indicate that it had not yet fully converged. Training for a higher number of epochs (e.g., 50+) would significantly enhance the image sharpness and further reduce artifacts.
*   **Hyperparameter Tuning:** With more computational resources, a more extensive grid search for hyperparameters (learning rate, batch size, and loss weights $\lambda_{rec} / \lambda_{id}$) could be conducted to optimize performance.
*   **Architecture Depth:** A deeper network with more residual blocks could be employed to capture more complex facial features, given better hardware.

##  How to Run

1.  **Clone the Repository:**
```bash
git clone https://github.com/YOUR_USERNAME/CelebA-Face-Enhancement-ML.git
cd CelebA-Face-Enhancement-ML
```

2. **Prerequisites:**
This project is implemented using **Python 3.x** and **PyTorch**. To replicate the results, ensure you have the following libraries installed.

It is highly recommended to use a virtual environment (conda or venv). You can install the dependencies via pip:
```bash
pip install numpy matplotlib opencv-python scikit-image torch torchvision tqdm
```
**Required Libraries:**
*   `torch` & `torchvision`: For building the U-Net architecture and Identity Loss computation.
*   `numpy` & `pandas`: For data manipulation.
*   `opencv-python` (`cv2`) & `scikit-image`: For image processing (degradation simulation).
*   `matplotlib`: For plotting training curves and visualizing results.
*   `tqdm`: For progress bars during training loops.

> **Note:** A system with an **NVIDIA GPU (CUDA)** is strongly recommended for training, although the code will run on CPU (significantly slower).

3. Running the Project
Open the main notebook file:
```bash
    jupyter notebook "p1 (1).ipynb"
```
Then run the cells sequentially.
