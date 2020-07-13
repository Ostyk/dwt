import depth_model
from ioUtils import *
import math
import lossFunction
import tensorflow as tf
import numpy as np
import sys
import os
import scipy.io as sio
import re
import time

VGG_MEAN = [103.939, 116.779, 123.68]

tf.set_random_seed(0)

def initialize_model(outputChannels, wd=None, modelWeightPaths=None):
    params = {"depth/conv1_1": {"name": "depth/conv1_1", "shape": [5,5,2,64], "std": None, "act": "relu", "reuse": False},
              "depth/conv1_2": {"name": "depth/conv1_2", "shape": [5,5,64,128], "std": None, "act": "relu", "reuse": False},
              "depth/conv2_1": {"name": "depth/conv2_1", "shape": [5,5,128,128], "std": None, "act": "relu", "reuse": False},
              "depth/conv2_2": {"name": "depth/conv2_2", "shape": [5,5,128,128], "std": None, "act": "relu", "reuse": False},
              "depth/conv2_3": {"name": "depth/conv2_3", "shape": [5,5,128,128], "std": None, "act": "relu", "reuse": False},
              "depth/conv2_4": {"name": "depth/conv2_4", "shape": [5,5,128,128], "std": None, "act": "relu", "reuse": False},
              "depth/fcn1": {"name": "depth/fcn1", "shape": [1,1,128,128], "std": None, "act": "relu", "reuse": False},
              "depth/fcn2": {"name": "depth/fcn2", "shape": [1,1,128,outputChannels], "std": None, "act": "relu", "reuse": False},
              "depth/upscore": {"name": "depth/upscore", "ksize": 8, "stride": 4, "outputChannels": outputChannels},
              }

    return depth_model.Network(params, wd=wd, modelWeightPaths=modelWeightPaths)

def forward_model(model, feeder, outputSavePath):
    with tf.Session() as sess:
        tfBatchDirs = tf.placeholder("float")
        tfBatchSS = tf.placeholder("float")
        keepProb = tf.placeholder("float")

        with tf.name_scope("model_builder"):
            print ("attempting to build model")
            model.build(tfBatchDirs, tfBatchSS, keepProb=keepProb)
            print ("built the model")

        init = tf.initialize_all_variables()

        sess.run(init)

        if not os.path.exists(outputSavePath):
            os.makedirs(outputSavePath)

        for i in range(int(math.floor(feeder.total_samples() / batchSize))):
            dirBatch, ssBatch, idBatch = feeder.next_batch()

            outputBatch = sess.run(model.outputDataArgMax, feed_dict={tfBatchDirs: dirBatch,
                                                                      tfBatchSS: ssBatch,
                                                                      keepProb: 1.0})
            outputBatch = outputBatch.astype(np.uint8)

            for j in range(len(idBatch)):
                outputFilePath = os.path.join(outputSavePath, idBatch[j]+'.mat')
                outputFileDir = os.path.dirname(outputFilePath)
                # print (outputFileDir)
                # print (outputFilePath)
                # raw_input("pause")

                if not os.path.exists(outputFileDir):
                    os.makedirs(outputFileDir)

                sio.savemat(outputFilePath, {"depth_map": outputBatch[j]})

                print ("processed image %d out of %d"%(j+batchSize*i, feeder.total_samples()))

def train_model(model, outputChannels, learningRate, trainFeeder, valFeeder, modelSavePath=None, savePrefix=None, initialIteration=1):
    with tf.Session() as sess:
        tfBatchDirs = tf.placeholder("float", shape=[None, 384, 384, 2])
        tfBatchGT = tf.placeholder("float", shape=[None, 384, 384])
        #tfBatchWeight = tf.placeholder("float")
        tfBatchSS = tf.placeholder("float", shape=[None, 384, 384])
        keepProb = tf.placeholder("float")

        with tf.name_scope("model_builder"):
            print ("attempting to build model")
            model.build(tfBatchDirs, tfBatchSS, keepProb=keepProb)
            print ("built the model")
        sys.stdout.flush()
        loss = lossFunction.modelTotalLoss(pred=model.outputData, gt=tfBatchGT, ss=tfBatchSS, outputChannels=outputChannels)
        #numPredictedWeighted = lossFunction.countTotalWeighted(ss=tfBatchSS)
        numPredicted = lossFunction.countTotal(ss=tfBatchSS)
        numCorrect = lossFunction.countCorrect(pred=model.outputData, gt=tfBatchGT, ss=tfBatchSS, k=1, outputChannels=outputChannels)

        train_op = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss=loss)

        #init = tf.initialize_all_variables()
        sess.run(tf.global_variables_initializer())

        iteration = initialIteration

        while iteration < 1000:
                batchLosses = []
                totalPredicted = 0
                totalCorrect = 0

                for k in range(int(math.floor(trainFeeder.total_samples() / batchSize))):
                    dirBatch, gtBatch, ssBatch = valFeeder.next_batch()

                    batchLoss, batchPredicted, batchCorrect = sess.run([loss, numPredicted, numCorrect],
                        feed_dict={tfBatchDirs: dirBatch,
                                   tfBatchGT: gtBatch,
                                   tfBatchSS: ssBatch,
                                   keepProb: 1.0})

                    batchLosses.append(batchLoss)
                    totalPredicted += batchPredicted
                    totalCorrect += batchCorrect

                if np.isnan(np.mean(batchLosses)):
                    print ("LOSS RETURNED NaN")
                    sys.stdout.flush()
                    print( 1)

                # print ("Itr: %d - b %d - val loss: %.3f, depth MSE: %.3f, exceed3: %.3f, exceed5: %.3f"%(iteration,j,
                #     float(np.mean(batchLosses)), totalDepthError/totalPredicted,
                #     totalExceed3/totalPredicted, totalExceed5/totalPredicted)
                print ("%s Itr: %d - val loss: %.6f, correct: %.6f" % (time.strftime("%H:%M:%S"),
                iteration, float(np.mean(batchLosses)), totalCorrect / totalPredicted))

                if (iteration > 0 and iteration % 5 == 0) or checkSaveFlag(modelSavePath):
                    modelSaver(sess, modelSavePath, savePrefix, iteration)

                    # print ("Processed iteration %d, batch %d" % (i,j)
                    # sys.stdout.flush()

                sys.stdout.flush()
                # raw_input("paused")
                #for j in range(10):
                for j in range(int(math.floor(valFeeder.total_samples() / batchSize))):
                    dirBatch, gtBatch, ssBatch = trainFeeder.next_batch()
                    sess.run(train_op, feed_dict={tfBatchDirs: dirBatch,
                                                  tfBatchGT: gtBatch,
                                                  tfBatchSS: ssBatch,
                                                  keepProb: 0.7})

                iteration += 1



def modelSaver(sess, modelSavePath, savePrefix, iteration, maxToKeep=5):
    allWeights = {}

    for name in [n.name for n in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]:
        param = sess.run(name)
        nameParts = re.split('[:/]', name)
        saveName = nameParts[-4]+'/'+nameParts[-3]+'/'+nameParts[-2]
        allWeights[saveName] = param

        # print ("Name: %s Mean: %.3f Max: %.3f Min: %.3f std: %.3f" % (name,
        #                                                              param.mean(),
        #                                                              param.max(),
        #                                                              param.min(),
        #                                                              param.std())
        # if name == "depth/fcn2/weights:0":
        #     for j in range(outputChannels):
        #         print ("ch: %d, max %e, min %e, std %e" % (
        #             j, param[:, :, :, j].max(), param[:, :, :, j].min(), param[:, :, :, j].std())

    # raw_input("done")

    sio.savemat(os.path.join(modelSavePath, savePrefix+'_%03d'%iteration), allWeights)


def checkSaveFlag(modelSavePath):
    flagPath = os.path.join(modelSavePath, 'saveme.flag')

    if os.path.exists(flagPath):
        return True
    else:
        return False

if __name__ == "__main__":
    
    outputChannels = 16
    classType = 'unified_CR'
    indices = [0]
    # 0=car, 1=person, 2=rider, 3=motorcycle, 4=bicycle, 5=truck, 6=bus, 7=train
    savePrefix = "depth_" + classType + "_CR_pretrain"

    train = True

    if train:
        batchSize = 4
        learningRate = 1e-5
        # learningRateActual = 1e-7
        wd = 1e-5
        initialIteration = 1

        model = initialize_model(outputChannels=outputChannels, wd=wd, modelWeightPaths=None)

        trainFeeder = Batch_Feeder(dataset_path="../../watershednet/data/for_training/42/", 
                           indices=indices,
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
                           indices=indices,
                           subset='val',
                           batchSize=batchSize,
                           padWidth=None,
                           padHeight=None, 
                           flip=False,
                           keepEmpty=False,
                           train=True,
                           img_shape = (384,384))
        valFeeder.set_paths()

        train_model(model=model, outputChannels=outputChannels,
            learningRate=learningRate,
            trainFeeder=trainFeeder, valFeeder=valFeeder,
            modelSavePath="../models/depth", savePrefix=savePrefix,
            initialIteration=initialIteration)

    else:
        batchSize = 5
        modelWeightPaths = []
        model = initialize_model(outputChannels=outputChannels, wd=None, modelWeightPaths=modelWeightPaths)

        feeder = Batch_Feeder(dataset="cityscapes", train=train, indices=indices, batchSize=batchSize, padWidth=None, padHeight=None)
        feeder.set_paths(idList=read_ids('./cityscapes/splits/vallist.txt'),
                            ssDir="./cityscapes/unified/ssMaskFineGT/val")

        forward_model(model, feeder=feeder,
                      outputSavePath="" % ())
