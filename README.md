# 🦴 **OsteoAI**  
## *Automatic classification of X-ray images in bone fractures or no fratures using Deep and Machine Learning*  

### **Project Structure**

``` markdown
OsteoAI/
├── App/
│    ├── app.py    ← Creates Streamlit app
│    ├── example_fracture.jpg
│    ├── example_nofracture.jpg
│    ├── lgbm.pkl  ← Model used for X-ray images classification (fracture vs non-fracture)
│    └── logo.png

├── data/
│   └── Processed/
│       └── ml-dp/  ← Structured folders for Deep Learning and Machine Learning models
│       │   ├── fracture/
│       │   └── nofracture/

├── Notebooks/
│   ├── 1-Create_directories.ipynb  ← Creates train/test/valid folders inside /ml-dp/
│   ├── 2-CNN_proofs.ipynb          ← Experiments with DL and ML models
│   └── 3-lgbm.proofs.ipynb         ← Experiments with Light Gradient-Boosting Model

├── Plots/

├── src/
│   ├── callback_training_CNN.py  ← Callbacks for CNN training
│   ├── copy_images.py            ← Copies images between directories
│   ├── creating.directories.py   ← Creates necessary directories
│   ├── extract_features.py       ← Extracts image features using CNN for ML model training
│   ├── image_loader.py           ← Loads and visualizes images
│   ├── image_preprocessor.py     ← Performs data augmentation and image normalization
│   ├── import_images.py          ← Creates datasets
│   └── metrics_CNN.py            ← Metrics for evaluating CNN performance (accuracy, loss, confusion matrix, AUC-ROC curve)

├── LICENSE

├── README.md

└── requirements.txt
```

### **Introduction**

**OsteoAI** is a **machine learning and deep learning** project focused on the **automatic classification of bone fractures from medical images**, specifically **X-ray images**. The project combines classical machine learning techniques with convolutional neural networks (CNN) to extract image features and classify the radiographs as either fracture or non-fracture.

This repository contains the complete pipeline, including:
- Data preprocessing and dataset organization for both binary and multi-class classification tasks.
- Training and evaluation of various machine learning and deep learning models.
- Integration with Streamlit for a user-friendly web interface.

**The goal of OsteoAI is to assist radiologists and medical professionals by providing accurate, explainable, and fast fracture detection predictions**.

### **App Usage Note**

The **Streamlit app** included in this project is optimized for use on **desktop or laptop devices**.
Mobile or tablet users may experience **layout issues** or reduced functionality due to resolution and compatibility limitations inherent to smaller screens.

To get the best experience:
- Use the latest version of Google Chrome or Mozilla Firefox.
- Ensure a stable internet connection for model loading and image uploads.
- Recommended screen resolution: 1280x720 or higher.

To launch the app locally:
``` python
cd App
streamlit run app.py
```


### **Materials and Methods**

**Data cleansing**
Data were obtained in (https://www.kaggle.com/), tittled *Bone Fracture Detection: Computer Vision Project* (Darabi 2024). For this project, only the file *BoneFractureYolo8* was used. The original dataset had eight different classes: non-fractured bone, humerus, humerus fracture, elbow positive, fingers positive, forearm fracture, shoulder fractures and wrist positive. 

Data cleansing was was performed to remove images with poor quality (e.g., overly dark or light images). Additionally, the data was restructured into two categories: fracture (which includes all images and labels associated with any type of fracture in the original dataset) and non-fracture.

**Fracture and non-fracture classification using CNNs**
All processing and modeling were done using Python 3.10.12 (Van Rossum & Drake, 2009). Before modeling a CNN, all images were loaded and preprocessed (rescaled and normalized). Libraries used included *TensorFlow* (Abadi et al., 2015), *OpenCV* (Bradski, 2000), *Keras* (Chollet et al., 2015), and *Matplotlib* (Hunter, 2007).

The dataset was split into three subsets: training (64%), validation (16%), and testing (20%) (see `1-Create_directories.ipynb` in the `Notebooks` folder).

*Tested models*:
- <u>Personalized CNNs</u>: Image size of 256x256 pixels
- <u>VGG16</u>: Image size of 224x224 pixels (Simonyan & Zisserman 2014; https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16)
- <u>ResNet-50</u>: Image size of 224x224 pixels (He et al 2015; ttps://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)


For pre-trained CNNs, **fine-tuning** and **transfer learning** were applied to adapt them to the new dataset. Additionally, CNNs were combined with machine learning classification algorithms (e.g., Random Forest, SVC, SGBoots, and LightGBM) to enhance results.  Models were trained using *Adam optimizer*, *binary cross-entropy loss*, and *accuracy* as the evaluation metric. The learning rate varied depending on the model. Two callbacks were used: *ReduceLROnPlateau* (to reduce the learning rate when the validation loss plateaued) and *EarlyStopping* (to avoid overfitting if the model did not improve after a certain number of epochs)(see `callbacks_training_CNN.py` in the `src` folder).

After training the models, they were evaluated using metrics such as the confusion matrix, precision, recall, and the AUC-ROC curve (see `metrics_CNN.py` in the `src` folder).

**Machine Learning Integration**
Image features were extracted using CNNs (see `extract_features.py` in the `src` folder) and used to train various machine learning classifiers, employing *Scikit-Learn* (Pedregosa et al., 2011).


### **Results**

The best performing model combined the **VGG16 neural network** to extract image features and the **Light Gradient-Boosting Model** for classification.


### **References**

    Abadi M, Agarwal A, Barham P, Brevdo E, Chen Z, ..., Zheng X. (2015). TensorFlow: Large-scale machine learning on heterogeneous systems. Software available from tensorflow.
org
    Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.
    Chollet, F., & others. (2015). Keras. https://keras.io
    Darabi PK. (2024). Bone Fracture Detection: A Computer Vision Project. DOI: 10.13140/RG.2.2 
14400.34569
    Harris CR, Millman KJ, Van der Walt SJ, Gommers R, Virtanen P, Cournapeau D, ..., Oliphant T
(2020) Array programming with NumPy. *Nature* 585: 357–362. DOI: 10.1038/s41586-020-2649-2. 
    He K, Zhang X, Ren S, Sun J. (2015). Deep Residual Learning for Image Recognition. *arXiv*. https://doi.org/10.48550/arXiv.1512.03385
    Hunter JD. (2007). Matplotlib: A 2D Graphics ENvironment. Computing in Science & Engineering 9(3)
: 90–95 Van Rossum G, Drake FL. (2009). Python 3 Reference Manual. Scotts Valley, CA: CreateSpace
    Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, ..., Duchesnay E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research* 12:
2825-2830
    Simonyan K, Zisserman A. (2014). Very Deep Convuntional Networks for Large-Scale Image Recognition. *arXiv*. https://doi.org/10.48550/arXiv.1409.1556
    Wang A, Chen H, Liu L, Chen K, Lin Z, Han J, Ding G. (2024). YOLOv10: Real-Time End-to-End 
Object Detection. DOI: 10.48550/arXiv.2405.14458


### **Aknowledgements**

I want to thank Christian Donaire (Nodd3r) for his supervision and support in this project, solving all my doubts and helping me in moments of blockage. Likewise, to my co-workers Andrés Baamonde and Jesús Campo, who have always found time to help me in this project and to solve problems more related to programming. On a personal level, to my partner Fer Cortés-Fossati for giving me emotional support throughout this master's degree and this project at times when I wanted to give it all up. And of course to my cats Mia, Kleo and Maki who, with their contagious joy, make everything more bearable.


### **License**

This project is licensed under the [MIT License](./LICENSE).


### **Author**

**Irene Martín Rodríguez**  
GitHub: [@irene-martin-rod](https://github.com/irene-martin-rod)  
LinkedIn: [Irene Martín Rodríguez](www.linkedin.com/in/irenemartin-rodriguez)
Email: martinrod.irene@gmail.com