# **OsteoAI**
## *Automatic detection of bone fractures*

THIS PROJECT IS UNDER CONSTRUCTION

### **Materials and Methods**
**Images and labels preprocessing**
Data were obtained in (https://www.kaggle.com/), tittled Bone Fracture Detection: Computer Vision Project (Darabi 2024). For this project, only the file *BoneFractureYolo8* was used. All the processing and modelling was maded using Python 3.11.9 (Van Rossum & Drake 2009). Images were reescale to the same size (640x640 Mpx) together with YOLO (You Only Look Once) labels, they were converted to a grayscale and normalized. For that, I used the libraries *Numpy* (Harris et al. 2020), *Matplotlib* (Hunter 2007) and *OpenCV* (Bradski 2000).

**Fracture detection**
A model selctions was first maded using *YOLOv10-Small*, *YOLOv10-Medium* and *YOLOv10-Balanced* models of *Ultralytics* library (Wang et al. 2024). I run these model using 100 epochs and a batch size of 16 images. 

### **References**
    Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.
    Darabi PK. (2024). Bone Fracture Detection: A Computer Vision Project. DOI: 10.13140/RG.2.2 
14400.34569
    Harris CR, Millman KJ, Van der Walt SJ, Gommers R, Virtanen P, Cournapeau D, ..., Oliphant T
(2020) Array programming with NumPy. Nature 585: 357–362. DOI: 10.1038/s41586-020-2649-2. 
    Hunter JD. (2007). Matplotlib: A 2D Graphics ENvironment. Computing in Science & Engineering 9(3)
: 90–95 Van Rossum G, Drake FL. (2009). Python 3 Reference Manual. Scotts Valley, CA: CreateSpace
    Wang A, Chen H, Liu L, Chen K, Lin Z, Han J, Ding G. (2024). YOLOv10: Real-Time End-to-End 
Object Detection. DOI: 10.48550/arXiv.2405.14458