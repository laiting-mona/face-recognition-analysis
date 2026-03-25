# PCA Face Recognition Analysis

This repository implements Principal Component Analysis (PCA) on the Olivetti faces dataset to analyze facial image reconstruction and principal components. The project demonstrates data centering, PCA decomposition, and image reconstruction using varying numbers of principal components.

## Project Overview

The analysis processes 401 grayscale face images (64x64 pixels) from a modified Olivetti dataset. Key tasks include computing the mean face, principal components ("principal faces"), and reconstructing the last face using 5, 20, 100, and 200 components. This showcases dimensionality reduction and visualization techniques in Python. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/47971400/40deee46-ba0c-4f86-9629-a3540783607e/faces_question-1.ipynb)

## Key Components

- **Data Loading**: Reads 401 faces from CSV (4096 pixels each).
- **Visualization**: Displays 16 random faces, mean face, first principal faces, and reconstructions.
- **PCA Implementation**: Centers data, extracts principal components (4096x401 matrix), computes projection coefficients.
- **Reconstruction**: Rebuilds images using top-k components to evaluate quality degradation. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/47971400/40deee46-ba0c-4f86-9629-a3540783607e/faces_question-1.ipynb)

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Scikit-learn (for PCA reference)

Install via pip:
```
pip install numpy matplotlib scikit-learn
```

## Usage

1. Place `faces.csv` in the repository root (replace `YOUR_PATH_TO_FACES_CSV`).
2. Run the notebook: `jupyter notebook faces_question-1.ipynb`.
3. Cells execute sequentially: imports, data display, PCA computation, and reconstruction plots. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/47971400/40deee46-ba0c-4f86-9629-a3540783607e/faces_question-1.ipynb)

## Results Structure

The notebook generates:

| Section | Output Description |
|---------|-------------------|
| Data Preview | 16 random 64x64 face images |
| PCA Components | Mean face + first 3 principal faces |
| Reconstruction | Last face rebuilt with 5, 20, 100, 200 components |

Expected outputs show progressive improvement in reconstruction fidelity as k increases. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/47971400/40deee46-ba0c-4f86-9629-a3540783607e/faces_question-1.ipynb)

## Code Highlights

Core PCA steps (user-completed sections):

```python
# Data centering
mu = np.mean(X, axis=1)  # 4096 x 1 mean face
XC = X - mu[:, np.newaxis]  # Centered data

# PCA (via SVD or sklearn)
pca = PCA()
pca.fit(XC.T)
A = pca.components_.T  # Principal components matrix

# Projection of last face
last_face = XC[:, 400]
Z = np.dot(A.T, last_face)  # 401 x 1 coefficients
```

Reconstruction formula: \( X_{recon} = Z[:k] \cdot A.T[:k, :] + \mu \). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/47971400/40deee46-ba0c-4f86-9629-a3540783607e/faces_question-1.ipynb)

## Limitations

- Requires local `faces.csv` (not included due to size).
- No model persistence or hyperparameter tuning.
- Focused on educational PCA implementation, not production face recognition.

## Author

Ting-Ying Lai  
National Taiwan University
