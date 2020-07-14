from model_io import *
import sys
from loss_function import *
import math
import time

def train_model(model, outputChannels, learningRate, trainFeeder, valFeeder,
                modelSavePath=None, savePrefix=None, initialIteration=1, batchSize=1):
    with tf.Session() as sess:
        tfBatchImages = tf.placeholder("float", shape=[None, 384, 384, 3])
        tfBatchGT = tf.placeholder("float", shape=[None, 384, 384])
        tfBatchSS = tf.placeholder("float", shape=[None, 384, 384])
        keepProb = tf.placeholder("float")

        with tf.name_scope("model_builder"):
            print ("attempting to build model")
            model.build(tfBatchImages, tfBatchSS, keepProb=keepProb)
            print ("built the model")
        sys.stdout.flush()


        loss = modelTotalLoss(pred=model.outputData, gt=tfBatchGT, ss=tfBatchSS, outputChannels=outputChannels)
        numPredicted = countTotal(ss=tfBatchSS)
        numCorrect = countCorrect(pred=model.outputData, gt=tfBatchGT, ss=tfBatchSS, k=1, outputChannels=outputChannels)

        print ("setting adam optimizer")
        sys.stdout.flush()

        train_op = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss=loss)

        init = tf.initialize_all_variables()
        print ("attempting to run init")
        sys.stdout.flush()

        sess.run(init)
        print ("completed init")
        sys.stdout.flush()

        iteration = initialIteration

        while iteration < 1000:
            batchLosses = []
            totalPredicted = 0
            totalCorrect = 0

            for k in range(int(math.floor(valFeeder.total_samples() / batchSize))):
                imageBatch, gtBatch, ssBatch = valFeeder.next_batch()

                batchLoss, batchPredicted, batchCorrect = sess.run(
                    [loss, numPredicted, numCorrect],
                    feed_dict={tfBatchImages: imageBatch,
                               tfBatchGT: gtBatch,
                               tfBatchSS: ssBatch,
                               keepProb: 1.0})

                batchLosses.append(batchLoss)
                totalPredicted += batchPredicted
                totalCorrect += batchCorrect

            if np.isnan(np.mean(batchLosses)):
                print ("LOSS RETURNED NaN")
                sys.stdout.flush()
                print(1) #change to return

            print ("%s Itr: %d - val loss: %.6f, correct: %.6f" % (time.strftime("%H:%M:%S"),
            iteration, float(np.mean(batchLosses)), totalCorrect / totalPredicted))
            sys.stdout.flush()

            if (iteration > 0 and iteration % 5 == 0) or checkSaveFlag(modelSavePath):
                modelSaver(sess, modelSavePath, savePrefix, iteration)

            #for j in range(10):
            for j in range(int(math.floor(trainFeeder.total_samples() / batchSize))):
                # print ("attempting to run train batch"
                # sys.stdout.flush()

                imageBatch, gtBatch, ssBatch = trainFeeder.next_batch()
                sess.run(train_op, feed_dict={tfBatchImages: imageBatch,
                                              tfBatchGT: gtBatch,
                                              tfBatchSS: ssBatch,
                                              keepProb: 0.7})
                # print ("ran one iteration"

            iteration += 1