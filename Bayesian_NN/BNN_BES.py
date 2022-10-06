####################################################################
# ---------------------------------------------------------------  #
# This Script loads root-Files of your choice, trains a            #
# Bayesian Neural Network with the provided Data and afterwards    #
# performs a Prediction on an unlabeled Dataset, you also have to  #
# provide in root Format.                                          #
# ---------------------------------------------------------------- #
# You can set the filenames yourself as well as which branches and #
# selection criteria to use. These can all be chosen differently   #
# for each Data Set you put in.                                    #
# ---------------------------------------------------------------- #
# It is of course possible to also change Training parameters but  #
# this has to be done in the main function.                        #
# ---------------------------------------------------------------- #
# For further questions please contact: timilittau@hotmail.de      #
# ---------------------------------------------------------------- #
####################################################################

# Here you can set the Name of your filenames, branches and the selection criteria

Signal     = 'SIM.root'
Background = 'EXP.root'
Predict    = 'EXP_CLEAN.root'

tName      = 'tLambdaTwoTrack'

branches   = ['TwoTrackLambdaP4.fP.fX', 'TwoTrackLambdaP4.fP.fY', 'TwoTrackLambdaP4.fP.fZ',
              'TwoTrackLambdaP4.fE', 'TwoTrackLambdaDecayLength',
              'TwoTrackLambdaVertex.fX', 'TwoTrackLambdaVertex.fY', 'TwoTrackLambdaVertex.fZ',
              'NTrack_X', 'NTrack_Y', 'NTrack_Z']

s_sel       = 'Chi2 > 0 && Chi2 < 5'
bg_sel      = 'Chi2 > 0'



# --------------------------------------------------------------------- #


# Import Routines for all packages needed

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import ROOT
from root_numpy import tree2array, root2array
from array import array
from random import randint

tfd = tfp.distributions

def importer(sFname, bgFname, rName, epoch_size, branches, S_selection, BG_selection, tName):
  # This is the input routine for all the data sets.
  # In the end all root-Data Sets are converted into suitable tensorflow compatible format
  
  np.random.seed(150)

  sFile = ROOT.TFile(sFname)
  bgFile = ROOT.TFile(bgFname)
  rFile = ROOT.TFile(rName)
  
  sTree = sFile.Get(tName)
  bgTree = bgFile.Get(tName)
  rTree = rFile.Get(tName)
  
  RsignalData = tree2array(sTree, branches=branches, selection=S_selection)
  RbgData = tree2array(bgTree, branches=branches, selection=BG_selection)
  RrData = tree2array(rTree, branches=branches)

  signalData = np.zeros((RsignalData.size, len(branches)+1))
  signalData[:,-1] = 1
  bgData = np.zeros((RbgData.size, len(branches)+1))
  rData = np.zeros((RrData.size, len(branches)+1))

  for j in range(len(branches)):
    for i in range(RsignalData.size):
      signalData[i,j] = RsignalData[i][j]
    for k in range(RbgData.size):
      bgData[k,j] = RbgData[k][j]
    for q in range(RrData.size):
      rData[q,j] = RrData[q][j]
  
  signalData = np.float32(signalData)
  bgData = np.float32(bgData)
  rData = np.float32(rData)


  np.random.shuffle(signalData)
  np.random.shuffle(bgData)

  training = np.concatenate((signalData[0:int(epoch_size),:], bgData[0:int(epoch_size),:]))
  validation = np.concatenate((signalData[int(epoch_size):int(2e5),:], bgData[int(epoch_size):int(2e5),:]))

  np.random.shuffle(training)
  np.random.shuffle(bgData)

  t_features = training[:,0:-1]
  t_labels = training[:,-1]

  v_features = validation[:,0:-1]
  v_labels = validation[:,-1]

  r_features = rData[:,0:-1]
  r_labels = rData[:,-1]

  t_dataset = tf.data.Dataset.from_tensor_slices((t_features, t_labels))
  v_dataset = tf.data.Dataset.from_tensor_slices((v_features, v_labels))
  r_dataset = tf.data.Dataset.from_tensor_slices((r_features, r_labels))

  h_size = len(v_labels)
  t_size = len(t_labels)
  r_size = len(r_labels)


  return t_dataset, v_dataset, t_size, h_size, r_dataset, r_size


def export_to_root(probs, fname, tName):
  # This Function saves the performed predictions of the unlabeled data into a root-file
  # It copies the content of the original unlabeled Data Set and creates a new file
  # with an additional branch called "Predictions"

  probabilities = probs[:,1]
  
  rFile = ROOT.TFile.Open(fname)
  events = rFile.Get(tName)

  Root_File = ROOT.TFile.Open("Predictions.root", "RECREATE")
  new_Tree = events.CloneTree()

  predictions = array( 'f', [0] )
  
  pred = Root_File.Get(tName)
  wei = Root_File.Get('Weights')

  p_branch = pred.Branch("Predictions", predictions, 'Predictions/F')
  
  
  for i in range(len(probabilities)):
    predictions[0] = probabilities[i]
    p_branch.Fill()


  Root_File.Write()
  Root_File.Close()
  print('Predictions saved to file')
  
  
def input_pipeline(training_dataset, heldout_dataset, batch_size, heldout_size, r_data, r_size):
  # This function creates all the tensorflow iterators and handles necessary to perform the training
    
  training_batches = training_dataset.shuffle(
    50000, reshuffle_each_iteration=True).repeat().batch(batch_size)
  training_iterator = training_batches.make_one_shot_iterator()

    
  heldout_frozen = (heldout_dataset.take(heldout_size).repeat().batch(heldout_size))
  heldout_iterator = heldout_frozen.make_one_shot_iterator()

  r_set = (r_data.take(r_size).repeat().batch(r_size))
  r_iterator = r_set.make_one_shot_iterator()

  handle = tf.placeholder(tf.string, shape=[])
  feedable_iterator = tf.data.Iterator.from_string_handle(
    handle, training_batches.output_types, training_batches.output_shapes)
  inputs, labels = feedable_iterator.get_next()

  return inputs, labels, handle, training_iterator, heldout_iterator, r_iterator

def main(sFname, bFname, rFname, branches, S_selection, BG_selection, tName):
  #del argv

  # This Program imports Data and executes a training with a final saving of the resulting Model

  # Setup of Parameters and filenames
  max_steps = 3000
  epoch_size = 1.5e5

  weight_file = 'weights.h5'

  print('Loading Dataset...')

  # Importing the Data from file and converting to tensorflow tensor
  (training, validation, training_size, heldout_size, rData, r_size) = importer(
    sFname, bFname, rFname, epoch_size, branches, S_selection, BG_selection, tName)

  print('Dataset Loaded!')

  # Setup of Learning Parameters

  learning_rate = 0.001
  batch_size = 100
  viz_steps = 100
  mc_num = 50
    

  print('Building Input Pipeline and NN')

  # Setup for the handles for the session

  (inputs, labels, handle, iterator, h_iterator, r_iterator) = input_pipeline(
    training, validation, batch_size, heldout_size, rData, r_size)

  # This is the actual Model
  with tf.name_scope("bayesian_neural_net", values=[inputs]):
    neural_net = tf.keras.Sequential([
      tfp.layers.DenseFlipout(22, activation=tf.nn.relu),
      tfp.layers.DenseFlipout(2)
    ])
    
    
    probs = neural_net(inputs)
    labels_distribution = tfd.Categorical(logits=probs)

  # Compute loss over batch_size
  neg_likelihood = -tf.reduce_mean(labels_distribution.prob(labels))
  kl = sum(neural_net.losses) / training_size
  elbo_loss = neg_likelihood + kl

  tf.summary.scalar('Loss', elbo_loss)

  # Evaluation of predictions
  predictions = tf.argmax(probs, axis=1)
  
  # Metrics for optimization in the training
  accuracy, accuracy_update_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

  tf.summary.scalar('accuracy', accuracy)

  # Extract weight posterior statistics for the weight layer
  names = []
  qmeans = []
  qstds = []
  
  for i, layer in enumerate(neural_net.layers):
    try:
      q = layer.kernel_posterior
    except AttributeError:
      continue
    names.append("Layer {}".format(i))
    qmeans.append(q.mean())
    qstds.append(q.stddev())

  

  # Set Optimizer
  with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(elbo_loss)

  # Simple Global Initializer
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())

  print('Everything build up! Starting Training...')
  
  # Start Session and begin training
  with tf.Session() as sess:
    # First, set up all handles for training, testing and prediction
    sess.run(init_op)
    
    train_handle = sess.run(iterator.string_handle())
    heldout_handle = sess.run(h_iterator.string_handle())
    p_handle = sess.run(r_iterator.string_handle())
    epoch = 1

    # This is the training-loop
    for step in range(max_steps):
      if (step-1) % (epoch_size/batch_size) == 0 and (step-1)>0:
        print('{:>2d}. Epoch done!'.format(epoch))
        epoch = epoch + 1
        
      _ = sess.run([train_op, accuracy_update_op],
                     feed_dict={handle: train_handle})
      

      # Every 100th step the loss and accuracy are printed out
      if step % 100 == 0:
        loss_value, accuracy_value = sess.run(
          [elbo_loss, accuracy], feed_dict={handle: train_handle})
        print("Step: {:>3d} Loss : {:.3f} Accuracy: {:.3f}".format(
          step, loss_value, accuracy_value)) 
        
      # Every "viz-step"th step an evaluation of the testing data is performed
      # within this step, the current weight state is saved
      if (step+1) % viz_steps == 0:
        
      # Compute probs of heldhout set
        probs = np.asarray([sess.run((labels_distribution.probs),
                                     feed_dict={handle: heldout_handle})
                            for _ in range(mc_num)])
        mean_probs = np.mean(probs, axis=0)
        input_vals, label_vals = sess.run((inputs, labels),
                                        feed_dict={handle: heldout_handle})
        accuracy_test = sess.run(accuracy, feed_dict={handle: heldout_handle})

        print("Step: {:>3d} Accuracy of Testing Data: {:.3f}".format(step, accuracy_test))

    # Compute Predictions on unlabeled Data Set    

    uk_probs = np.asarray([sess.run((labels_distribution.probs),
                                   feed_dict={handle: p_handle})
                          for _ in range(mc_num)])
    muk_probs = np.mean(uk_probs, axis=0)
    uk_input, _ = sess.run((inputs, labels),
                           feed_dict = {handle: p_handle})

    # Save Model and performed predictions
    neural_net.save_weights(weight_file)

    export_to_root(muk_probs, rFname, tName)




    
main(sFname = Signal, bFname = Background, rFname = Predict,
      branches = branches, S_selection = s_sel, BG_selection = bg_sel,
      tName=tName)
    
