# CNN-based-Wafer-Defect-Patterns-Recognition-with-Variational-Auto-Encoder
This repository is implementation of CNN-based-Wafer-Defect-Patterns-Recognition-with-Variational-Auto-Encoder.
The purpose is classifying defects on wafers, which are key components in the semiconductor industry.
The CNNs, an image classification machine learning technique, and VAE to solve data imbalance were used.



## Dataset
[WM-811K wafer map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)

*If your goal is to check the reproducibility of this code, you can use preprocessed data in the 'Dataset' directory.



## Parser
| Parser name                | Option             | Help                                   |
|----------------------------|--------------------|----------------------------------------|
| Preprocessed_data_using    | True / False       | Whether to use preprocessed data       |
| VAE_lr                     | Float value        | VAE model learning rate                | 
| VAE_epochs                 | Int value          | VAE model training epochs              |
| VAE_batch_size             | Int value          | VAE model batch size                   |
| Network                    | MLP / CNN1 / CNN2  | Which network to use for classifying   |
| Model_lr                   | Float value        | Network model learning rate            |
| Model_epochs               | Int value          | Network model training epochs          |
| Model_batch_size           | Int value          | Netowrk model batch size               |



## Results
Our model achieves the following performance on: 

| Network  | Best accuracy (%) | Accuracy (%)   |
|----------|-------------------|----------------|
| MLP      | 95.16             | 95.10 ± 0.05   |
| CNN1     | 96.43             | 96.27 ± 0.13   |
| CNN2     | 99.14             | 99.07 ± 0.05   |

*MLP : 128-128-9   
*CNN1 : 64C5-MP2-128C5-MP2-9  
*CNN2 : 64C5-128C5-MP2-256C5-MP2-256-9  


