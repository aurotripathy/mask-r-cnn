# Training a Mask R-CNN model to find defects in Xray Images

This project trains a Mask R-CNN model written in Keras/TensorFlow to find defects in Xray Images. 

It reuses all of the code available [here](https://github.com/matterport/Mask_RCNN) (additionally, refactoring for modularity). Training script available at 

`mask-r-cnn/samples/gdxray/run_train.sh` Hit me up if you want the mask-annotated training data to improve the results. 

Sample results are shown below. Each instance of a predicted mask is color-coded. The dashed bounding-box (in green) is the bounding-box ground-truth. 

<p float="left">
  <img src="samples/gdxray/results/val/C0001/C001.01/C001.png" width="425" />
  <img src="samples/gdxray/results/val/C0021/C0021_0025/C0021_0025_BB.png" width="425" /> 
</p>
<p float="left">
  <img src="samples/gdxray/results/val/C0026/C0026_0010_BB/C0026_0010_BB.png" width="425" />
  <img src="samples/gdxray/results/val/C0030/C0030_0009/C0030_0009_BB.png" width="425" /> 
</p>
<!--
![mark-r-cnn](samples/gdxray/results/val/C0001/C001.01/C001.png) 
![mark-r-cnn](samples/gdxray/results/val/C0021/C0021_0025/C0021_0025_BB.png)
![mark-r-cnn](samples/gdxray/results/val/C0026/C0026_0010_BB/C0026_0010_BB.png)
![mark-r-cnn](samples/gdxray/results/val/C0030/C0030_0009/C0030_0009_BB.png)
-->

All the of customization needed to train and evaluate the model is in in the folder, `mask-r-cnn/samples/gdxray/`

