# tf-tcd
I downgraded my Anaconda to use python 3.5. For that

```
conda install python=3.5
```

Next, I ran src/readtcd.py, with proper paths.

Then I ran
```
python object_detection/train.py --pipeline_config_path=$HOME/temp/tf-tcd/config/ssd_mobilenet_v1_coco-tcd.config --train_dir=$HOME/temp/miotcd-dataset/MIO-TCD-Localization/train/ --logtostderr 
```
