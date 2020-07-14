from network_init import get_model
from io_utils import *
import tensorflow as tf
from forward import forward_model
from train import train_model
import os

tf.set_random_seed(0)

if __name__ == "__main__":
    outputChannels = 16
    savePrefix = "model"
    outputPrefix = ""
    # 0=car, 1=person, 2=rider, 3=motorcycle, 4=bicycle, 5=truck, 6=bus, 7=train
    train = True

    if train:
        batchSize = 3
        learningRate = 5e-6 # usually i use 5e-6
        wd = 1e-6

        #modelWeightPaths = [""]
        model_dir = "../best_models"
        modelWeightPaths = [os.path.join(model_dir, i) for i in os.listdir(model_dir)]

        initialIteration = 1
        trainFeeder = Batch_Feeder(dataset_path="../../watershednet/data/for_training/42/", 
                           unet_output_path = '../../pytorch-nested-unet/outputs/42',
                           subset='train',
                           batchSize=batchSize,
                           padWidth=None,
                           padHeight=None, 
                           flip=False,
                           keepEmpty=False,
                           train=True,
                           img_shape = (384,384))
        trainFeeder.set_paths()

        valFeeder = Batch_Feeder(dataset_path="../../watershednet/data/for_training/42/", 
                                 unet_output_path = '../../pytorch-nested-unet/outputs/42',
                                   subset='val',
                                   batchSize=batchSize,
                                   padWidth=None,
                                   padHeight=None, 
                                   flip=False,
                                   keepEmpty=False,
                                   train=True,
                                   img_shape = (384,384))
        valFeeder.set_paths()

        model = get_model(wd=wd, modelWeightPaths=modelWeightPaths)

        train_model(model=model, outputChannels=outputChannels,
                    learningRate=learningRate,
                    trainFeeder=trainFeeder,
                    valFeeder=valFeeder,
                    modelSavePath="../models/E2E",
                    savePrefix=savePrefix,
                    initialIteration=initialIteration)

    else:
        batchSize = 1
        modelWeightPaths = ["../model/dwt_cityscapes_pspnet.mat"]

        model = get_model(modelWeightPaths=modelWeightPaths)

        feeder = Batch_Feeder(dataset="cityscapes",
                                      train=train,
                                      batchSize=batchSize)

        feeder.set_paths(idList=read_ids('../example/sample_list.txt'),
                         imageDir="../example/inputImages",
                            ssDir="../example/PSPNet")

        forward_model(model, feeder=feeder,
                      outputSavePath="../example/output")
