# tcd-tf

## Set up Tensorflow
Downgrade Anaconda to use python 3.5. The reason for this are the non-compliance of Tensorflow to Anaconda3.6. For that do:

```
conda install python=3.5
pip install tensorflow tensorflow-gpu
```

Clone the Tensorflow repo:
```
git clone --recursive https://github.com/tensorflow/models.git
```

Do not forget to compile the protobuf in the research folder:
```
cd models
protoc object_detection/protos/*.proto --python_out=.
```

## Preparing the files
Clone this repo:
```
cd && git clone git@github.com:tokudaek/tf-tcd.git
```

Run src/readtcd.py, with proper paths.


Adjust the paths of the config file. If the relative paths are the same, just run:
```
sed -i 's#$HOME#'"$HOME#g" config/MYMODEL.config
```

## Training
Then do:
```
cd $TENSORFLOWMODELS/research
python object_detection/train.py --pipeline_config_path=$HOME/tf-tcd/config/ssd_mobilenet_v1_coco-tcd.config --train_dir=$HOME/temp/20171206-train_miotcd --logtostderr   
```

## Exporting the generated model
From `tensorflow/models/research/`
```
python object_detection/export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path <PIPELINE CONFIG> \
--trained_checkpoint_prefix <CHECKPOINT PREFIX> \
--output_directory /tmp/output-model
```

## Evaluation

Evaluate the generated model.
```
export PYTHONPATH=$HOME/projects/tf-models/research/:$HOME/projects/tf-models/research/object_detection/:$HOMEprojects/tf-models/research/slim
```
