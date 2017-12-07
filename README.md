# tf-tcd
I downgraded my Anaconda to use python 3.5. For that

```
conda install python=3.5
```

Do not forget to compile the protobuf in research folder:
```
protoc object_detection/protos/*.proto --python_out=.
```
Next, I ran src/readtcd.py, with proper paths.

Then I ran
```
python object_detection/train.py --pipeline_config_path=$HOME/temp/tf-tcd/config/ssd_mobilenet_v1_coco-tcd.config --train_dir=$HOME/temp/20171206-train_miotcd --logtostderr   
```

Adjust the paths of the config file:
```
sed -i 's#$HOME#'"$HOME#g" config/MYMODEL.config
```
