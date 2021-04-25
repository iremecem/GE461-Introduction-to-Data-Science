# -*- coding: utf-8 -*-
"""
Author: Irem Ecem Yelkanat
"""

import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sammon import sammon

"""### Load the Dataset and Split the Data Into Half as Train and Test Data"""

# Load dataset
data = loadmat("digits.mat")
features = data["digits"]
labels = data["labels"]

# Split data into half between train and test
train_X, test_X, train_Y, test_Y = train_test_split(features, labels, test_size=0.5, random_state=42, shuffle=True)

"""## Apply PCA"""

# Apply PCA
pca = PCA(n_components=400)
pca.fit_transform(train_X)
eigen_values = pca.explained_variance_
principal_components = pca.components_
pves = pca.explained_variance_ratio_

"""### Plot the Eigenvalues in Descending Order"""

# Plot Principal Components vs. Eigenvalues
plt.plot([x for x in range(1, 401)] ,eigen_values)
plt.title("Principal Component Order vs. Eigenvalue")
plt.xlabel("Principal Component Order")
plt.ylabel("Eigenvalue")
plt.show()

# Plot Principal Components vs. Proportion of Explained Variance
plt.plot([x for x in range(1, 401)], pves)
plt.title("Principal Component Order vs. PVE")
plt.xlabel("Principal Component Order")
plt.ylabel("Proportion of Explained Variance")
plt.show()

"""### Display the Mean of the Training Data"""

# Find the mean of data 
train_X_mean = (pca.mean_.reshape(20,20)).T

# Display mean of data as colored
plt.imshow(train_X_mean)
plt.title("Mean of Training Data - Colored")
plt.axis("off")
plt.show()

# Display mean of data as gray
plt.imshow(train_X_mean, cmap="gray")
plt.title("Mean of Training Data - Gray")
plt.axis("off")
plt.show()

"""### Display First 100 Eigenvectors"""

# Display eigenvectors - first 100
first_75_pc = plt.figure()
first_75_pc.suptitle("First 75 Principal Components")
for index in range(75):
    pc = (principal_components[index]).reshape(20, 20)
    first_75_pc.add_subplot(10, 10, index + 1)
    plt.imshow(pc.T, cmap="gray") # show the image as black & white
    plt.axis('off') # turn off axis ticks
plt.show(block=True)

"""### Create Different Subspaces, Project Data and Train Gaussian Classifier"""

# Initialize variables for storing the data
component_count = [i for i in range(2, 201, 2)]
classification_errors_train = []
classification_errors_test = []

for i in component_count:

    # Get i number of components and fit training data
    pca = PCA(n_components=i, random_state=42)
    pca.fit(train_X)

    # Project train and test data to principal components
    train_X_projected = pca.transform(train_X)
    test_X_projected = pca.transform(test_X)

    # Create model and fit train data
    model = GaussianNB()
    model.fit(train_X_projected, (train_Y.T)[0])

    # Predict output for Train data
    predict_test_X = model.predict(train_X_projected)
    accuracy_train = metrics.accuracy_score((train_Y.T)[0], predict_test_X)
    classification_error_train = 1 - accuracy_train
    classification_errors_train.append(classification_error_train)

    # Predict output for Test data
    predict_test_Y = model.predict(test_X_projected)
    accuracy_test = metrics.accuracy_score((test_Y.T)[0], predict_test_Y)
    classification_error_test = 1 - accuracy_test
    classification_errors_test.append(classification_error_test)

"""### Plot Classification Error vs. the Number of Components Used for Each Subspace"""

# Plot error rate on training data
plt.plot(component_count, classification_errors_train)
plt.xlabel("Component Count")
plt.ylabel("Error Rate")
plt.title("Error Rate on Train Data - PCA")
plt.show()

# Plot error rate on test data
plt.plot(component_count, classification_errors_test)
plt.xlabel("Component Count")
plt.ylabel("Error Rate")
plt.title("Error Rate on Test Data - PCA")
plt.show()

"""## Apply LDA"""

# Apply LDA
lda = LDA(n_components=9)
lda.fit(train_X, (train_Y.T)[0])
scalings = lda.scalings_

"""### Display New Bases"""

# Display new bases
lda_bases = plt.figure()
lda_bases.suptitle("LDA Bases")
for index in range(9):
    base = (scalings[:,index]).reshape(20, 20)
    lda_bases.add_subplot(2, 5, index + 1)
    plt.imshow(base, cmap="gray") # show the image as black & white
    plt.axis('off') # turn off axis ticks
plt.show(block=True)

"""### Create Different Subspaces, Project Data and Train Gaussian Classifier"""

# Initialize variables for storing the data
component_count = [i for i in range(1, 10)]
classification_errors_train = []
classification_errors_test = []

for i in component_count:

    # Get i number of components and fit training data
    lda = LDA(n_components=i)
    lda.fit(train_X, (train_Y.T)[0])

    # Project train and test data to principal components
    train_X_projected = lda.transform(train_X)
    test_X_projected = lda.transform(test_X)

    # Create model and fit train data
    model = GaussianNB()
    model.fit(train_X_projected, (train_Y.T)[0])

    # Predict output for Train data
    predict_test_X = model.predict(train_X_projected)
    accuracy_train = metrics.accuracy_score((train_Y.T)[0], predict_test_X)
    classification_error_train = 1 - accuracy_train
    classification_errors_train.append(classification_error_train)

    # Predict output for Test data
    predict_test_Y = model.predict(test_X_projected)
    accuracy_test = metrics.accuracy_score((test_Y.T)[0], predict_test_Y)
    classification_error_test = 1 - accuracy_test
    classification_errors_test.append(classification_error_test)

"""### Plot Classification Error vs. the Dimension of Each Subspace"""

# Plot error rate on training data
plt.plot(component_count, classification_errors_train)
plt.xlabel("Dimension of the Subspace")
plt.ylabel("Error Rate")
plt.title("Error Rate on Train Data - LDA")
plt.show()

# Plot error rate on test data
plt.plot(component_count, classification_errors_test)
plt.xlabel("Dimension of the Subspace")
plt.ylabel("Error Rate")
plt.title("Error Rate on Test Data - LDA")
plt.show()

"""# Sammon's Mapping"""

# [y_250,E] = sammon(features, n=2, maxiter=250, maxhalves=20)

[y_500,E] = sammon(features, n=2, maxiter=500, maxhalves=20)

# [y,E] = sammon(features, n=2, maxiter=1000, maxhalves=20)

# Plot
# plt.scatter(y_250[labels[:,0] == 0, 0], y_250[labels[:,0] == 0, 1], s=5, marker='o',label="0")
# plt.scatter(y_250[labels[:,0] == 1, 0], y_250[labels[:,0] == 1, 1], s=5, marker='o',label="1")
# plt.scatter(y_250[labels[:,0] == 2, 0], y_250[labels[:,0] == 2, 1], s=5, marker='o',label="2")
# plt.scatter(y_250[labels[:,0] == 3, 0], y_250[labels[:,0] == 3, 1], s=5, marker='o',label="3")
# plt.scatter(y_250[labels[:,0] == 4, 0], y_250[labels[:,0] == 4, 1], s=5, marker='o',label="4")
# plt.scatter(y_250[labels[:,0] == 5, 0], y_250[labels[:,0] == 5, 1], s=5, marker='o',label="5")
# plt.scatter(y_250[labels[:,0] == 6, 0], y_250[labels[:,0] == 6, 1], s=5, marker='o',label="6")
# plt.scatter(y_250[labels[:,0] == 7, 0], y_250[labels[:,0] == 7, 1], s=5, marker='o',label="7")
# plt.scatter(y_250[labels[:,0] == 8, 0], y_250[labels[:,0] == 8, 1], s=5, marker='o',label="8")
# plt.scatter(y_250[labels[:,0] == 9, 0], y_250[labels[:,0] == 9, 1], s=5, marker='o',label="9")
# plt.title('Sammon\'s Mapping for Digits Data with 250 Iterations')
# plt.legend(loc="upper left", bbox_to_anchor=(1,1), title="Digits")
# plt.show()

# Plot
# plt.scatter(y[labels[:,0] == 0, 0], y[labels[:,0] == 0, 1], s=5, marker='o',label="0")
# plt.scatter(y[labels[:,0] == 1, 0], y[labels[:,0] == 1, 1], s=5, marker='o',label="1")
# plt.scatter(y[labels[:,0] == 2, 0], y[labels[:,0] == 2, 1], s=5, marker='o',label="2")
# plt.scatter(y[labels[:,0] == 3, 0], y[labels[:,0] == 3, 1], s=5, marker='o',label="3")
# plt.scatter(y[labels[:,0] == 4, 0], y[labels[:,0] == 4, 1], s=5, marker='o',label="4")
# plt.scatter(y[labels[:,0] == 5, 0], y[labels[:,0] == 5, 1], s=5, marker='o',label="5")
# plt.scatter(y[labels[:,0] == 6, 0], y[labels[:,0] == 6, 1], s=5, marker='o',label="6")
# plt.scatter(y[labels[:,0] == 7, 0], y[labels[:,0] == 7, 1], s=5, marker='o',label="7")
# plt.scatter(y[labels[:,0] == 8, 0], y[labels[:,0] == 8, 1], s=5, marker='o',label="8")
# plt.scatter(y[labels[:,0] == 9, 0], y[labels[:,0] == 9, 1], s=5, marker='o',label="9")
# plt.title('Sammon\'s Mapping for Digits Data with 1000 Iterations')
# plt.legend(loc="upper left", bbox_to_anchor=(1,1), title="Digits")
# plt.show()

# Plot
plt.scatter(y_500[labels[:,0] == 0, 0], y_500[labels[:,0] == 0, 1], s=5, marker='o',label="0")
plt.scatter(y_500[labels[:,0] == 1, 0], y_500[labels[:,0] == 1, 1], s=5, marker='o',label="1")
plt.scatter(y_500[labels[:,0] == 2, 0], y_500[labels[:,0] == 2, 1], s=5, marker='o',label="2")
plt.scatter(y_500[labels[:,0] == 3, 0], y_500[labels[:,0] == 3, 1], s=5, marker='o',label="3")
plt.scatter(y_500[labels[:,0] == 4, 0], y_500[labels[:,0] == 4, 1], s=5, marker='o',label="4")
plt.scatter(y_500[labels[:,0] == 5, 0], y_500[labels[:,0] == 5, 1], s=5, marker='o',label="5")
plt.scatter(y_500[labels[:,0] == 6, 0], y_500[labels[:,0] == 6, 1], s=5, marker='o',label="6")
plt.scatter(y_500[labels[:,0] == 7, 0], y_500[labels[:,0] == 7, 1], s=5, marker='o',label="7")
plt.scatter(y_500[labels[:,0] == 8, 0], y_500[labels[:,0] == 8, 1], s=5, marker='o',label="8")
plt.scatter(y_500[labels[:,0] == 9, 0], y_500[labels[:,0] == 9, 1], s=5, marker='o',label="9")
plt.title('Sammon\'s Mapping for Digits Data with 500 Iterations')
plt.legend(loc="upper left", bbox_to_anchor=(1,1), title="Digits")
plt.show()

# Plot only 0
# plt.scatter(y_250[labels[:,0] == 0, 0], y_250[labels[:,0] == 0, 1], s=5, marker='o')
# plt.title('Sammon\'s Mapping for Digit 0 with 250 Iterations')
# plt.show()

# Plot only 0
plt.scatter(y_500[labels[:,0] == 0, 0], y_500[labels[:,0] == 0, 1], s=5, marker='o')
plt.title('Sammon\'s Mapping for Digit 0 with 500 Iterations')
plt.show()

# Plot only 0
# plt.scatter(y[labels[:,0] == 0, 0], y[labels[:,0] == 0, 1], s=5, marker='o')
# plt.title('Sammon\'s Mapping for Digit 0 with 1000 Iterations')
# plt.show()

# # Plot only 1
# plt.scatter(y_250[labels[:,0] == 1, 0], y_250[labels[:,0] == 1, 1], s=5, c='gold', marker='o')
# plt.title('Sammon\'s Mapping for Digit 1 with 250 Iterations')
# plt.show()

# Plot only 1
plt.scatter(y_500[labels[:,0] == 1, 0], y_500[labels[:,0] == 1, 1], s=5, c='gold', marker='o')
plt.title('Sammon\'s Mapping for Digit 1 with 500 Iterations')
plt.show()

# Plot only 1
# plt.scatter(y[labels[:,0] == 1, 0], y[labels[:,0] == 1, 1], s=5, c='gold', marker='o')
# plt.title('Sammon\'s Mapping for Digit 1 with 1000 Iterations')
# plt.show()

# # Plot only 2
# plt.scatter(y_250[labels[:,0] == 2, 0], y_250[labels[:,0] == 2, 1], s=5, c='limegreen', marker='o')
# plt.title('Sammon\'s Mapping for Digit 2 with 250 Iterations')
# plt.show()

# Plot only 2
plt.scatter(y_500[labels[:,0] == 2, 0], y_500[labels[:,0] == 2, 1], s=5, c='limegreen', marker='o')
plt.title('Sammon\'s Mapping for Digit 2 with 500 Iterations')
plt.show()

# Plot only 2
# plt.scatter(y[labels[:,0] == 2, 0], y[labels[:,0] == 2, 1], s=5, c='limegreen', marker='o')
# plt.title('Sammon\'s Mapping for Digit 2 with 1000 Iterations')
# plt.show()

# # Plot only 3
# plt.scatter(y_250[labels[:,0] == 3, 0], y_250[labels[:,0] == 3, 1], s=5, c='deeppink', marker='o')
# plt.title('Sammon\'s Mapping for Digit 3 with 250 Iterations')
# plt.show()

# Plot only 3
plt.scatter(y_500[labels[:,0] == 3, 0], y_500[labels[:,0] == 3, 1], s=5, c='deeppink', marker='o')
plt.title('Sammon\'s Mapping for Digit 3 with 500 Iterations')
plt.show()

# # Plot only 3
# plt.scatter(y[labels[:,0] == 3, 0], y[labels[:,0] == 3, 1], s=5, c='deeppink', marker='o')
# plt.title('Sammon\'s Mapping for Digit 3 with 1000 Iterations')
# plt.show()

# # Plot only 4
# plt.scatter(y_250[labels[:,0] == 4, 0], y_250[labels[:,0] == 4, 1], s=5, c='plum', marker='o')
# plt.title('Sammon\'s Mapping for Digit 4 with 250 Iterations')
# plt.show()

# Plot only 4
plt.scatter(y_500[labels[:,0] == 4, 0], y_500[labels[:,0] == 4, 1], s=5, c='plum', marker='o')
plt.title('Sammon\'s Mapping for Digit 4 with 500 Iterations')
plt.show()

# # Plot only 4
# plt.scatter(y[labels[:,0] == 4, 0], y[labels[:,0] == 4, 1], s=5, c='plum', marker='o')
# plt.title('Sammon\'s Mapping for Digit 4 with 1000 Iterations')
# plt.show()

# # Plot only 5
# plt.scatter(y_250[labels[:,0] == 5, 0], y_250[labels[:,0] == 5, 1], s=5, c='brown', marker='o')
# plt.title('Sammon\'s Mapping for Digit 5 with 250 Iterations')
# plt.show()

# Plot only 5
plt.scatter(y_500[labels[:,0] == 5, 0], y_500[labels[:,0] == 5, 1], s=5, c='brown', marker='o')
plt.title('Sammon\'s Mapping for Digit 5 with 500 Iterations')
plt.show()

# # Plot only 5
# plt.scatter(y[labels[:,0] == 5, 0], y[labels[:,0] == 5, 1], s=5, c='brown', marker='o')
# plt.title('Sammon\'s Mapping for Digit 5 with 1000 Iterations')
# plt.show()

# # Plot only 6
# plt.scatter(y_250[labels[:,0] == 6, 0], y_250[labels[:,0] == 6, 1], s=5, c='hotpink', marker='o')
# plt.title('Sammon\'s Mapping for Digit 6 with 250 Iterations')
# plt.show()

# Plot only 6
plt.scatter(y_500[labels[:,0] == 6, 0], y_500[labels[:,0] == 6, 1], s=5, c='hotpink', marker='o')
plt.title('Sammon\'s Mapping for Digit 6 with 500 Iterations')
plt.show()

# # Plot only 6
# plt.scatter(y[labels[:,0] == 6, 0], y[labels[:,0] == 6, 1], s=5, c='hotpink', marker='o')
# plt.title('Sammon\'s Mapping for Digit 6 with 1000 Iterations')
# plt.show()

# # Plot only 7
# plt.scatter(y_250[labels[:,0] == 7, 0], y_250[labels[:,0] == 7, 1], s=5, c='slategray', marker='o')
# plt.title('Sammon\'s Mapping for Digit 7 with 250 Iterations')
# plt.show()

# Plot only 7
plt.scatter(y_500[labels[:,0] == 7, 0], y_500[labels[:,0] == 7, 1], s=5, c='slategray', marker='o')
plt.title('Sammon\'s Mapping for Digit 7 with 500 Iterations')
plt.show()

# # Plot only 7
# plt.scatter(y[labels[:,0] == 7, 0], y[labels[:,0] == 7, 1], s=5, c='slategray', marker='o')
# plt.title('Sammon\'s Mapping for Digit 7 with 1000 Iterations')
# plt.show()

# # Plot only 8
# plt.scatter(y_250[labels[:,0] == 8, 0], y_250[labels[:,0] == 8, 1], s=5, c='yellowgreen', marker='o')
# plt.title('Sammon\'s Mapping for Digit 8 with 250 Iterations')
# plt.show()

# Plot only 8
plt.scatter(y_500[labels[:,0] == 8, 0], y_500[labels[:,0] == 8, 1], s=5, c='yellowgreen', marker='o')
plt.title('Sammon\'s Mapping for Digit 8 with 500 Iterations')
plt.show()

# # Plot only 8
# plt.scatter(y[labels[:,0] == 8, 0], y[labels[:,0] == 8, 1], s=5, c='yellowgreen', marker='o')
# plt.title('Sammon\'s Mapping for Digit 8 with 1000 Iterations')
# plt.show()

# # Plot only 9
# plt.scatter(y_250[labels[:,0] == 9, 0], y_250[labels[:,0] == 9, 1], s=5, c='cyan', marker='o')
# plt.title('Sammon\'s Mapping for Digit 9 with 250 Iterations')
# plt.show()

# Plot only 9
plt.scatter(y_500[labels[:,0] == 9, 0], y_500[labels[:,0] == 9, 1], s=5, c='cyan', marker='o')
plt.title('Sammon\'s Mapping for Digit 9 with 500 Iterations')
plt.show()

# # Plot only 9
# plt.scatter(y[labels[:,0] == 9, 0], y[labels[:,0] == 9, 1], s=5, c='cyan', marker='o')
# plt.title('Sammon\'s Mapping for Digit 9 with 1000 Iterations')
# plt.show()

"""# t-SNE"""

tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=5000, random_state=42)
tsne_results = tsne.fit_transform(features)

# Plot
plt.scatter(tsne_results[labels[:,0] == 0, 0], tsne_results[labels[:,0] == 0, 1], s=5, marker='o',label="0")
plt.scatter(tsne_results[labels[:,0] == 1, 0], tsne_results[labels[:,0] == 1, 1], s=5, marker='o',label="1")
plt.scatter(tsne_results[labels[:,0] == 2, 0], tsne_results[labels[:,0] == 2, 1], s=5, marker='o',label="2")
plt.scatter(tsne_results[labels[:,0] == 3, 0], tsne_results[labels[:,0] == 3, 1], s=5, marker='o',label="3")
plt.scatter(tsne_results[labels[:,0] == 4, 0], tsne_results[labels[:,0] == 4, 1], s=5, marker='o',label="4")
plt.scatter(tsne_results[labels[:,0] == 5, 0], tsne_results[labels[:,0] == 5, 1], s=5, marker='o',label="5")
plt.scatter(tsne_results[labels[:,0] == 6, 0], tsne_results[labels[:,0] == 6, 1], s=5, marker='o',label="6")
plt.scatter(tsne_results[labels[:,0] == 7, 0], tsne_results[labels[:,0] == 7, 1], s=5, marker='o',label="7")
plt.scatter(tsne_results[labels[:,0] == 8, 0], tsne_results[labels[:,0] == 8, 1], s=5, marker='o',label="8")
plt.scatter(tsne_results[labels[:,0] == 9, 0], tsne_results[labels[:,0] == 9, 1], s=5, marker='o',label="9")
plt.title("t-SNE with Perplexity 15, 2000 Iterations and Random State 13")
plt.legend(loc="upper left", bbox_to_anchor=(1,1), title="Digits")
plt.show()

tsne = TSNE(n_components=2, verbose=1, perplexity=15, n_iter=2000, random_state=13)
tsne_results = tsne.fit_transform(features)

# Plot
plt.scatter(tsne_results[labels[:,0] == 0, 0], tsne_results[labels[:,0] == 0, 1], s=5, marker='o',label="0")
plt.scatter(tsne_results[labels[:,0] == 1, 0], tsne_results[labels[:,0] == 1, 1], s=5, marker='o',label="1")
plt.scatter(tsne_results[labels[:,0] == 2, 0], tsne_results[labels[:,0] == 2, 1], s=5, marker='o',label="2")
plt.scatter(tsne_results[labels[:,0] == 3, 0], tsne_results[labels[:,0] == 3, 1], s=5, marker='o',label="3")
plt.scatter(tsne_results[labels[:,0] == 4, 0], tsne_results[labels[:,0] == 4, 1], s=5, marker='o',label="4")
plt.scatter(tsne_results[labels[:,0] == 5, 0], tsne_results[labels[:,0] == 5, 1], s=5, marker='o',label="5")
plt.scatter(tsne_results[labels[:,0] == 6, 0], tsne_results[labels[:,0] == 6, 1], s=5, marker='o',label="6")
plt.scatter(tsne_results[labels[:,0] == 7, 0], tsne_results[labels[:,0] == 7, 1], s=5, marker='o',label="7")
plt.scatter(tsne_results[labels[:,0] == 8, 0], tsne_results[labels[:,0] == 8, 1], s=5, marker='o',label="8")
plt.scatter(tsne_results[labels[:,0] == 9, 0], tsne_results[labels[:,0] == 9, 1], s=5, marker='o',label="9")
plt.title("t-SNE with Perplexity 15, 2000 Iterations and Random State 13")
plt.legend(loc="upper left", bbox_to_anchor=(1,1), title="Digits")
plt.show()

tsne = TSNE(n_components=2, verbose=1, perplexity=23, n_iter=2000, random_state=42)
tsne_results = tsne.fit_transform(features)

# Plot
plt.scatter(tsne_results[labels[:,0] == 0, 0], tsne_results[labels[:,0] == 0, 1], s=5, marker='o',label="0")
plt.scatter(tsne_results[labels[:,0] == 1, 0], tsne_results[labels[:,0] == 1, 1], s=5, marker='o',label="1")
plt.scatter(tsne_results[labels[:,0] == 2, 0], tsne_results[labels[:,0] == 2, 1], s=5, marker='o',label="2")
plt.scatter(tsne_results[labels[:,0] == 3, 0], tsne_results[labels[:,0] == 3, 1], s=5, marker='o',label="3")
plt.scatter(tsne_results[labels[:,0] == 4, 0], tsne_results[labels[:,0] == 4, 1], s=5, marker='o',label="4")
plt.scatter(tsne_results[labels[:,0] == 5, 0], tsne_results[labels[:,0] == 5, 1], s=5, marker='o',label="5")
plt.scatter(tsne_results[labels[:,0] == 6, 0], tsne_results[labels[:,0] == 6, 1], s=5, marker='o',label="6")
plt.scatter(tsne_results[labels[:,0] == 7, 0], tsne_results[labels[:,0] == 7, 1], s=5, marker='o',label="7")
plt.scatter(tsne_results[labels[:,0] == 8, 0], tsne_results[labels[:,0] == 8, 1], s=5, marker='o',label="8")
plt.scatter(tsne_results[labels[:,0] == 9, 0], tsne_results[labels[:,0] == 9, 1], s=5, marker='o',label="9")
plt.title("t-SNE with Perplexity 23, 2000 Iterations and Random State 42")
plt.legend(loc="upper left", bbox_to_anchor=(1,1), title="Digits")
plt.show()

tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=2000, random_state=49)
tsne_results = tsne.fit_transform(features)

# Plot
plt.scatter(tsne_results[labels[:,0] == 0, 0], tsne_results[labels[:,0] == 0, 1], s=5, marker='o',label="0")
plt.scatter(tsne_results[labels[:,0] == 1, 0], tsne_results[labels[:,0] == 1, 1], s=5, marker='o',label="1")
plt.scatter(tsne_results[labels[:,0] == 2, 0], tsne_results[labels[:,0] == 2, 1], s=5, marker='o',label="2")
plt.scatter(tsne_results[labels[:,0] == 3, 0], tsne_results[labels[:,0] == 3, 1], s=5, marker='o',label="3")
plt.scatter(tsne_results[labels[:,0] == 4, 0], tsne_results[labels[:,0] == 4, 1], s=5, marker='o',label="4")
plt.scatter(tsne_results[labels[:,0] == 5, 0], tsne_results[labels[:,0] == 5, 1], s=5, marker='o',label="5")
plt.scatter(tsne_results[labels[:,0] == 6, 0], tsne_results[labels[:,0] == 6, 1], s=5, marker='o',label="6")
plt.scatter(tsne_results[labels[:,0] == 7, 0], tsne_results[labels[:,0] == 7, 1], s=5, marker='o',label="7")
plt.scatter(tsne_results[labels[:,0] == 8, 0], tsne_results[labels[:,0] == 8, 1], s=5, marker='o',label="8")
plt.scatter(tsne_results[labels[:,0] == 9, 0], tsne_results[labels[:,0] == 9, 1], s=5, marker='o',label="9")
plt.title("t-SNE with Perplexity 5, 2000 Iterations and Random State 49")
plt.legend(loc="upper left", bbox_to_anchor=(1,1), title="Digits")
plt.show()

