# IoT2BE

IoT Sensory Data to Business Event Prediction(IoT2BE)

 ![image](https://user-images.githubusercontent.com/86541762/221352946-e7fd8d0b-cf2d-475a-9ad6-f19907c5930b.png)
This is a Pytorch implementation of IoT2BE.

![image](https://user-images.githubusercontent.com/86541762/221353867-a963a5f0-7e0e-4502-8242-22f0debe46bb.png)
This is the transformation and feature extraction of business events.

# Motivating Examples
![image](https://user-images.githubusercontent.com/86541762/221353970-7e084636-9081-4d7f-a523-586561ae58b5.png)



# Datasets

The dataset file is located in the "dataset" directory.
The file format is .npy, which can be read using numpy in Python.
The dataset has a 3-dimensional shape of [n,m,k], where n represents the number of business event records, m represents the number of business events in each record, and k represents the number of feature attributes for each business event.
