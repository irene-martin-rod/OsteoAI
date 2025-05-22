# <div align="center"> **OsteoAI** </div> 
## <div align="center"> *Automatic classification of X-ray images in bone fractures or no fratures using Deep and Machine Learning* </div> 

⚠️ **This is a theoretical prototype. Not for medical use**


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


### **App Usage Note**

You can find the **Streamlit app** in the next link https://osteoai.streamlit.app/.

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


### **Abstract**
**OsteoAI** is a **machine and deep learning-based tool** for the **automatic detection of bone fractures in medical images**. Fast and accurate fracture identification is critical in emergency and clinical settings, yet often subject to human error or time constraints. This project proposes a solution that leverages convolutional neural networks trained on radiographic images to assist in fracture detection. After thorough data preprocessing and model training, the system achieves strong performance metrics. OsteoAI is designed to be scalable, open-source, and applicable in real-world medical workflows.


### **Introduction**

Fractures are among the most common reasons for emergency room visits (Tanzi et al., 2020). Statistics show that there are nearly three million bone fractures annually in France, Germany, Italy, Spain, Sweden, and the UK alone (Tanzi et al., 2020). The overall incidence is 11.67 per 1,000 inhabitants for men and 10.65 per 1,000 for women each year (Yang et al., 2020). Many patients suffer long-term consequences due to undiagnosed or untreated fractures (Tanzi et al., 2020). Therefore, early and accurate detection of bone fractures is critical.

Nonetheless, misdiagnoses by radiologists still occur, often due to a combination of factors such as fatigue from excessive workloads, emergency situations, night shifts (Hallas et al., 2006; Yang et al., 2020), the inherent limitations of human vision (Yang et al., 2020), lack of experience (Tanzi et al., 2020), or the subtle nature of certain fractures (Tanzi et al., 2020).

In recent years, artificial intelligence (AI) in medical image processing has attracted increasing attention (Su et al., 2023), particularly the application of deep learning (Su et al., 2023). The integration of deep learning with traditional diagnostic methods has given rise to a new field known as computational radiology (Meena & Roy, 2022). Deep learning not only enhances diagnostic accuracy but also helps reduce the workload of radiologists (Meena & Roy, 2022). Academic research has demonstrated that deep learning models can, in some cases, outperform human doctors in diagnostic tasks (Lindsey et al., 2018). Architectures such as ResNet, DenseNet, and EfficientNet have achieved high accuracy on benchmark datasets like MURA (Rajpurkar et al., 2017) and the RSNA Bone Fracture Detection dataset (https://www.kaggle.com/).

As a result, the number of studies exploring deep learning for fracture detection has steadily increased. Recent research has focused on detecting specific types of fractures or fractures in specific bones, such as wrist fractures (see Thian et al., 2019; Raisuddin et al., 2021; Joshi et al., 2022; Gan et al., 2024; Hasen et al., 2024), hip fractures (see Badgeley et al., 2019; Cheng et al., 2019; Krogue et al., 2020; Gao et al., 2023; Kim et al., 2024), or humerus fractures (see Chung et al., 2018; Kekatpure et al., 2024; Spek et al., 2024), among others.

OsteoAI is a machine learning and deep learning project focused on the automatic classification of bone fractures from medical images, specifically X-rays. The project combines classical machine learning techniques with convolutional neural networks (CNNs) to extract image features and classify radiographs as either fracture or non-fracture. OsteoAI aims to support radiologists and medical professionals by providing an automatic fracture detection system based on deep convolutional networks.


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

1. Abadi M, Agarwal A, Barham P, Brevdo E, Chen Z, ..., Zheng X. (2015). TensorFlow: Large-scale machine learning on heterogeneous systems. Software available from tensorflow. org
2. Badgeley MA, Zech JR, Oakden-Rayner L, Glicksberg B, Liu M, Gale W, ..., Dudley JT. (2019). Deep learning predicts hip fracture using confounding patient and healthcare variables. *NPJ digital medicine* 2(1): 31. https://doi.org/10.1038/s41746-019-0105-1
3. Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.
4. Cheng CT, Ho TY, Lee TY, Chang CC, Chou CC, Chen CC, ..., Liao CH. (2019). Application of a deep learning algorithm for detection and visualization of hip fractures on plain pelvic radiographs. *Europian Radiolpgy* 29: 5469–5477. https://doi.org/10.1007/s00330-019-06167-y
5. Chollet, F., & others. (2015). Keras. https://keras.io
5. Chung SW, Han SS, Lee JW, Oh KS, Kim NR, Yoon JP, ..., Kim Y. (2018).Automated detection and classification of the proximal humerus fracture by using deep learning algorithm. *Acta Orthopaedica* 89(4), 468–473. https://doi.org/10.1080/17453674.2018.1453714
6. Darabi PK. (2024). Bone Fracture Detection: A Computer Vision Project. DOI: 10.13140/RG.2.2 14400.34569
7. Gan K, Liu Y, Zhang T, Xu D, Lian L, Luo Z, ..., Lu L. (2024). Deep Learning Model for Automatic Identification and Classification of Distal Radius Fracture. *Journal of Imaging Informatics in Medicine*: 37(6): 2874-2882. doi: 10.1007/s10278-024-01144-4.
8. Gao Y, Soh NYT, Liu N, Lim G, Ting D, Cheng LTE, ..., Yan YY. (2023). Application of a deep learning algorithm in the detection of hip fractures. *Iscience*: 26(8): 107350
9. Hallas P, Ellingsen T. (2006) Errors in fracture diagnoses in the emergency department – characteristics of patients and diurnal variation. *BMC emergency medicine* 6: 1-5. https://doi.org/10.1186/1471-227X-6-4
9. Hansen V, Jensen J, Kusk MW, Gerke O, Tromborg HB, Lysdahlgaard S. (2024). Deep learning performance compared to healthcare experts in detecting wrist fractures from radiographs: A systematic review and meta-analysis. *European Journal of Radiology* 174: 111399. https://doi.org/10.1016/j.ejrad.2024.111399.
10. Harris CR, Millman KJ, Van der Walt SJ, Gommers R, Virtanen P, Cournapeau D, ..., Oliphant T. (2020). Array programming with NumPy. *Nature* 585: 357–362. DOI: 10.1038/s41586-020-2649-2. 
11. He K, Zhang X, Ren S, Sun J. (2015). Deep Residual Learning for Image Recognition. *arXiv*. https://doi.org/10.48550/arXiv.1512.03385
12. Hunter JD. (2007). Matplotlib: A 2D Graphics ENvironment. Computing in Science & Engineering 9(3): 90–95 Van Rossum G, Drake FL. (2009). Python 3 Reference Manual. Scotts Valley, CA: CreateSpace
13. Joshi D, Singh TP, Joshi AK. (2022). Deep learning-based localization and segmentation of wrist fractures on X-ray radiographs. *Neural Computing and Applications* 34: 19061–19077. https://doi.org/10.1007/s00521-022-07510-z
14. Kekatpure A, Kekatpure A, Deshpande S, Srivastava S. (2024). Development of a diagnostic support system for distal humerus fracture using artificial intelligence. *International Orthopaedics (SICOT)* 48: 1303–1311. https://doi.org/10.1007/s00264-024-06125-4
14. Kim Y , Kim YG, Park JW, Kim BW, Shin Y, Kong SH, ..., Shin CS. (2024). A CT-based Deep Learning Model for Predicting Subsequent Fracture Risk in Patients with Hip Fracture. *Radiology* 310: 1
15. Krogue JD, Cheng KV, Hwang KM, Toogood P, Meinberg EG, Geiger EJ, ..., Pedoia V. (2020 )Automatic Hip Fracture Identification and Functional Subclassification with Deep Learning. *Radiology: Artificial Intelligence* 2(2): e190023.
16. Lindsey R, Daluiski A, Chopra S, Lachapelle A, Mozer M, ..., Potter H. (2018). Deep neural network improves fracture detection by clinicians. *Proceedings of the National Academy of Sciences* 115(45): 11591–11596. https://doi.org/10.1073/pnas.1806905115
16. Meena T, Roy S. (2022). Bone Fracture Detection Using Deep Supervised Learning from Radiological Images: A Paradigm Shift. *Diagnosis* 12(10): 2420. https://doi.org/10.3390/diagnostics12102420
17. Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, ..., Duchesnay E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research* 12:
2825-2830
18. Rajpurkar P, Irvin J, Bagul A, Ding D, Duan T, Mehta H, ..., Ng AY. (2017). MURA: Large dataset for abnormality detection in musculoskeletal radiographs. *arXiv:1712.06957*. https://arxiv.org/abs/1712.06957. https://doi.org/10.48550/arXiv.1712.06957
19. Raisuddin AM, Vaattovaara E, Nevalainen M, Nikki M, Järvenpää E, Makkonen K, ..., Tiulpin A. (2021). Critical evaluation of deep neural networks for wrist fracture detection. *Scintific Reports* 11: 6006. https://doi.org/10.1038/s41598-021-85570-2
20. Simonyan K, Zisserman A. (2014). Very Deep Convuntional Networks for Large-Scale Image Recognition. *arXiv*. https://doi.org/10.48550/arXiv.1409.1556
22. Spek RWA, Smith WJ, Sverdlov M, Broos S, Zhao Y, Liao Z, ..., Doornberg JN. (2024). Detection, classification, and characterization of proximal humerus fractures on plain radiographs. *The Bone & Joint Journal*, 106-B(11), 1348-1360. https://doi.org/10.1302/0301-620X.106B11.BJJ-2024-0264.R1
21. Su Z, Adam A, Nasrudin MF, Ayob M, Punganan G. (2023). Skeletal fracture detection with deep learning: A comprehensive review. *Diagnostics* 13(20): 3245. https://doi.org/10.3390/diagnostics13203245
22. Tanzi L, Vezzetti E, Moreno R, Moos S. (2020). X-Ray Bone Fracture Classification Using Deep Learning: A Baseline for Designing a Reliable Approach. *Applied Science* 10(4): 1507. https://doi.org/10.3390/app10041507
23. Thian YL, Li Y, Jagmohan P, Sia D, Chang, VEY, Tan RT. (2019). Convolutional Neural Networks for Automated Fracture Detection and Localization on Wrist Radiographs. *Radiology: Artificial Intelligence* 1(1): e180001.
24. Wang A, Chen H, Liu L, Chen K, Lin Z, Han J, Ding G. (2024). YOLOv10: Real-Time End-to-End Object Detection. DOI: 10.48550/arXiv.2405.14458
25. Yang S, Yin B, Feng C, Fan G, He S (2020). Diagnostic accuracy of deep learning in orthopaedic fractures: a systematic review and meta-analysis. *Clincal Radiology* 75(9): 713-728. https://doi.org/10.1016/j.crad.2020.05.021


### **Aknowledgements**

I want to thank Christian Donaire (Nodd3r) for his supervision and support in this project, solving all my doubts and helping me in moments of blockage. Likewise, to my co-workers Andrés Baamonde and Jesús Campo, who have always found time to help me in this project and to solve problems more related to programming. On a personal level, to my partner Fer Cortés-Fossati for giving me emotional support throughout this master's degree and this project at times when I wanted to give it all up. And of course to my cats Mia, Kleo and Maki who, with their contagious joy, make everything more bearable.


### **License**

This project is licensed under the [MIT License](./LICENSE).


### **Author**

**Irene Martín Rodríguez**  
GitHub: [@irene-martin-rod](https://github.com/irene-martin-rod)

LinkedIn: [Irene Martín Rodríguez](www.linkedin.com/in/irenemartin-rodriguez)

Email: martinrod.irene@gmail.com














https://pubmed.ncbi.nlm.nih.gov/29269036/