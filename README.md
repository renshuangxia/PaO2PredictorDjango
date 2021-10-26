# PaO2PredictorDjango
Simple web page loads pretrained models for predicting PaO2 values

## Environment set up:
To set up the conda environment, use req.txt 

## Examples of PaO2 Predition

### 3 Features:

SpO2 = 95  , FiO2 = 0.5 ,  Peep = 5

|                 |  Samples with SpO2 <= 96    |  All Samples |
|:---------------:|:----------:|:------------:|
|Linear Regression|   111.95   |    141.28    |
|     SVR         |   93.44    |     95.5     |
| Neural Network  |   110.48   |    112.29    |


### 7 Features:

SpO2 = 95, FiO2 = 0.5, Peep = 10, Tidal Volumn = 850, MAP = 84, Temperature = 36.56, vaso = 0

|                 |  Samples with SpO2 <= 96   |  All Samples |
|:---------------:|:--------:|:------------:|
|Linear Regression|  100.07  |    138.53   |
|     SVR         |   86.79  |     85.12    |
| Neural Network  |   93.93  |     101.6    |