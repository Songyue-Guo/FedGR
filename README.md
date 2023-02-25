# FedGR :Federated Learning with Gravitation Regulation for Double Imbalance Distribution
federated learning for double unbalance settings (sample quantities imbalance for different classes in client and label or class imbalance for different client cross-client)
## Framework of FedGR
![Framework of FedGR](https://github.com/Guosy-wxy/FedGR/blob/main/images/framework.png)
## Part of Experiment Results (full results are listed in our paper)
 | Algorithms      |        CIFAR-10 (2)        |       CIFAR-10 (3)        |       CIFAR-100 (20)       |      {CIFAR-100 (30)       |
| --------------- | :------------------------: | :-----------------------: | :------------------------: | :------------------------: |
|                 |           Acc(%)           |          Acc(%)           |           Acc(%)           |           Acc(%)           |
| FedAvg          |           50.36            |           53.76           |           36.15            |           42.19            |
| FedProx         |           48.84            |           54.94           |           36.24            |           42.21            |
| FedNova         |           56.33            |           68.63           |           38.63            |           45.35            |
| SCAFFOLD        |           57.37            |           67.32           |           38.43            |           46.82            |
| PerFedAvg       |           44.67            |           54.87           |           35.98            |           40.14            |
| pFedMe          |           45.81            |           50.18           |           35.36            |           40.18            |
| FedOpt          |           62.37            |           70.63           |           42.37            |           49.63            |
| MOON            |           61.45            |           70.45           |           40.53            |           47.91            |
| FedRS           |        <u>63.22</u>        |       <u>73.56</u>        |        <u>42.76</u>        |           50.73            |
| FedGC           |           62.91            |           72.11           |           42.11            |           50.21            |
| **FedGR(ours)** | **67.84 ** | **77.86 ** | **45.44 ** | **53.16 ** |
## quick start 
```python
python main_fed.py -algo fedgr -dataset cifar10
```
## Citation
This is the code for the 2023 DASFAA paper: FedGR: Federated Learning with Gravitation Regulation for Double Imbalance Distribution. Please cite our paper if you use the code or datasets:
```
@inproceedings{Guo2023FedGR
  author    = {Songyue Guo and
               Xu Yang and
               Jiyuan Feng and
               Ye Ding and 
               Wei Wang and
               Yunqing Feng and
               Qing Liao},
  title     = {FedGR: Federated Learning with Gravitation Regulation for Double Imbalance Distribution
},
  booktitle = {Database Systems for Advanced Applications - 28th International Conference,
               {DASFAA} 2023, Tianjin, China, April 17-20, 2023},
  publisher = {Springer},
  year      = {2023}
}
```
