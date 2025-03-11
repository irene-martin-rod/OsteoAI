# **OsteoAI**
## *Automatic detection of bone fractures*

THIS PROJECT IS UNDER CONSTRUCTION

### **Project estructure**
``` markdown
/OSTEOAI/
|-- /data/

    |-- /Processed/ 

        |-- /BoneFractureYolo8/ <-- Here, it is saved all image and labels after quality screening but maintaining the original labels of the dataset (8 categories)
            |-- /test/
                |-- /images/
                |-- /labels/
            |-- /train/
                |-- /images/
                |-- /labels/
            |-- /valid/
                |-- /images/
                |-- /labels/
            |-- data.yaml <-- Configuration file
            |-- README.dataset.txt

        |-- /ml-dp/ <-- Folder structurized for DL and ML algorithms
            |-- /fracture/ <-- Contains images classified as fracture
            |-- /nofracture/ <-- Contains images classified as non-fracture

        |-- /Yolo-binary/ <-- Folder structurized for Yolo, but only has 2 categories (fractuve vs. non-fracture)
            |-- /images/
            |-- /labels/
            |-- data.yaml

|-- /Notebooks/
    |-- 1-Create_directories.ipynb <-- Create the directories train, test and valid in the folder /ml-dp/
    |-- 2-CNN_proofs.ipynb <-- Notebooks with all DL and ML models

|-- /Plots/

|-- /scr/
    │-- callback_training_CNN.py
    |-- copy_images.py
    |-- creating.directories.py
    |-- extract_features.py
    |-- image_loader.py
    |-- image_preporcesser.py
    |-- import_images.py
    |-- lgb_Vgg16.py
    |-- metrics_CNN.py

|-- LICENSE
|-- README.md
|-- requirements.txt
```


### **Materials and Methods**
**Data cleansing**
Data were obtained in (https://www.kaggle.com/), tittled Bone Fracture Detection: Computer Vision Project (Darabi 2024). For this project, only the file *BoneFractureYolo8* was used. The original dataset had eight different classes: non-fractured bone, humerus, humerus fracture, elbow positive, fingers positive, forearm fracture, shoulder fractures and wrist positive. A data cleansing was carried out, deleting images with poor quality, in other words, very dark or light images. Also, data were restructures in two categories: fracture (this category collects all images and labels associated with any type of fracture in the original dataset) and non-fracture.

**Fracture and non-fracture classification using CNNs**
All the processing and modelling was maded using Python 3.10.12 (Van Rossum & Drake 2009). Before modelling a CNN, all images were loaded and preprocessed: they were rescaling and normalized (see import_image.py, image_loader.py and image_preprocesser.py in scr folder) using the libraries *TensorFlow* (Abadi et al 2015), *OpenCV* (Bradski 2000) and *Matplotlib* (Hunter 2007). The imagen size depended on the used CNN. 

Dataset were divided in three subdataset: train (64% of images of dataset), validation (16% of images of the dataset) and test (20% of images of dataset)(see 1-Create_directories.ipynb in Notebooks folders). Several CNNs were proved: a CNN built from scratch using an imagen size of 256x256 Mpx, *VGG16* neural network (Simonyan & Zisserman 2014; https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16), using an image size of 224x224 Mpx, and ResNet-50 neural network (He et al 2015; ttps://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50), using an image size of 224x224 Mpx. All CNNs were import and/or training and validation using the library *Keras* (Chollet et al 2015). 

In the case of pre-trainning CNNs, I used fine-tunning and transfer-learning to train them to the new dataset. Also, I used a combination of CNN and Machine learning classification algorithims. For that, image feature were extracted (see extract_features.py in scr folder) using the CNN and then, these features were used for training a Machine learning algorithm, such as random forest and SVC, using the library Scikit-Learn (Pedregosa et al 2011). To run models, I used *Adam* as optimizer, *binary cross-entropy* as loss function and *accuracy* as metric. The learning rate was variable depend on the model. Also, I used two callbacks: *ReduceLROnPlateau* to reduece learning rate if the validation loss function was not reduce each certain epochs and *EarlyStopping* to avoid overfitting if the model did not improve after some epochs (see callbacks_training_CNN.py in scr folder). 

After running a model, I plotted the accuracy and loss function hostory of training and validation dataset using *Matplotlib* (Hunter 2007). In case both metrics were good, I obtained the confusion matrix (with metrics such as recall, precision and F1-score) and ROC curve. Both metrics were used to compare the best model among the better models (see CNN_proofs.ipynb in the Notebooks folder). 

After model selection explained above, I selected as the best model the combination of the *VGG16* neural network and LightGBM (AUC = 0.8737). 

**Fracture detection**
Images were loaded and preprocessing, reescaling all images to the same size (the size depending of the model) and normalized.  

and Images were reescale to the same size (640x640 Mpx) together with YOLO (You Only Look Once) labels, they were converted to a grayscale and normalized. For that, I used the libraries *Numpy* (Harris et al. 2020), *Matplotlib* (Hunter 2007) and *OpenCV* (Bradski 2000).

A model selctions was first maded using *YOLOv10-Small*, *YOLOv10-Medium* and *YOLOv10-Balanced* models of *Ultralytics* library (Wang et al. 2024). I run these model using 100 epochs and a batch size of 16 images. 

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