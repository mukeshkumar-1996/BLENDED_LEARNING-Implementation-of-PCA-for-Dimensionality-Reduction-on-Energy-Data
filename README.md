# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: V MUKESHKUMAR
RegisterNumber: 25012063 
*/
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('HeightsWeights.csv')
print("First 5 rows of the dataset:")
print(data.head())
X=data[['Height(Inches)', 'Weight(Pounds)']]
plt.figure(figsize=(6,5))
sns.scatterplot(x='Height(Inches)', y='Weight(Pounds)', data=data)
plt.title("Original Data Distribution")
plt.show()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
plt.figure(figsize=(6,5))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

## Output:
<img width="728" height="163" alt="Screenshot 2026-03-17 202443" src="https://github.com/user-attachments/assets/bd48061d-b4c5-4ed8-aa68-99e5ff788598" />

<img width="790" height="582" alt="Screenshot 2026-03-17 202452" src="https://github.com/user-attachments/assets/27a1f76e-0de1-4acd-ae21-295f3d41ae10" />
<img width="723" height="598" alt="Screenshot 2026-03-17 202504" src="https://github.com/user-attachments/assets/cdde3561-2dec-4ffa-8d7a-1f660a471382" />


## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
