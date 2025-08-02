# ACP-FTNet: Fusion of Transformer and Bi-LSTM with CNN for Anticancer Peptide Prediction via Comprehensive Feature Learning
The identification and characterization of anticancer peptides (ACPs) has emerged as a critical research area, as they hold the potential to revolutionize cancer therapy by offering more targeted and less toxic alternatives to traditional treatments. Traditional biological experimental methods for identifying ACPs are often costly, time consuming, and inefficient, posing significant challenges for rapid detection.
In this study, we introduce a new model, a novel framework for the prediction of anticancer peptides that employs a comprehensive feature extraction strategy, integrating amino acid composition and physicochemical properties. By combining representations using convolutional neural networks with those based on Transformer encoding layers, the model can capture diverse features of peptides better and more quickly. Its architecture utilizes bidirectional long-short-term memory (Bi-LSTM) networks, leveraging their strengths in capturing local sequence patterns and long-range dependencies.
We tested the new model on six public datasets and the experimental results demonstrate that the new model exhibits strong robustness and is more competitive than the compared methods, particularly in terms of AUC and ACC.
# How do we use ACP-FTNet?
## Environment setup 
Firstly, you need to create a virtual environment and set python==3.8
```
conda create --name yourEnv python=3.8
conda activate yourEnv
```
Then, if you want to run this model, please install package in environment.txt file
```
pip install -r environment.txt
```
Finally, we could utilize the ACP-FTNet model.
