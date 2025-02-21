# A Fast and Robust Geometric Point Cloud Registration Model for Orthopedic Surgery with Noisy and Incomplete Data

<p align="center">
<img src="https://github.com/xzh128/PER/blob/main/GFR.png">
</p>

 **This repository contains python scripts for Geometric Fast Registration(GFR)** <br>
 We propose the GFR model, which is noise-resistant, fast, and robust, suitable for defective and partially overlapping point sets. Experiments across clinical datasets demonstrate the versatility and effectiveness of our model. GFR consists of the following three models: <br>
 * PER operates within the frequency domain to enhance point cloud data by attenuating noise and reconstructing incomplete regions. <br>
* DAT augments feature representation by correlating independent features from source and target pointclouds, improving model expressiveness. <br>
 * GFM identifies geometrically consistent point pairs, completing missing data and refining registration accuracy. <br>
 ## Configuration
 This code is based on PyTorch implementation，and the python requirements are: <br>
 CUDA 11.7 <br>
 Pytorch 2.1.0 <br>
 Python 3.10 <br>
 More requirements can be found in requirements.txt <br>
 
 
