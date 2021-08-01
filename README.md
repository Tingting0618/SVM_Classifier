# SVM and KNN Classifier

#### Content Includes:
- Support Vector Machine (SVM) classifier
  - Linear SVM Classification
    - hard margin classification (not working well when there are outliers) 
    - soft margin classification (not working well when classes are not linearly separable)
  - Nonlinear SVM Classification
    - One approach to handling nonlinear datasets is to add more features, such as polynomial features; in some cases this can result in a linearly separable dataset.
    - That said, at a low polynomial degree, this method cannot deal with very complex datasets, and with a high polynomial degree it creates a huge number of features, making the model too slow.
  - Fortunately, the kernel trick makes it possible to get the same result as if you had added many polynomial features, even with very high-degree polynomials, without actually having to add them.
    - Polynomial Kernel
    - Gaussian RBF Kernel 
- KNeighbors (KNN) classifier

## Learn More

For more information, please check out the [Project Portfolio](https://tingting0618.github.io).

## Reference

This repo is my learning journal following:
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, by Aurélien Géron (O’Reilly). Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.
