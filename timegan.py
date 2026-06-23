"""Time-series Generative Adversarial Networks (TimeGAN).

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

# Necessary Packages
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import time
from utils import extract_time, rnn_cell, random_generator, batch_generator


def timegan(ori_data, parameters, n_generate=None):
  """TimeGAN function.
  
  Use original data as training set to generater synthetic data (time-series)
  
  Args:
    - ori_data: original time-series data
    - parameters: TimeGAN network parameters
    - n_generate: number of synthetic samples to generate (default: same as ori_data)
    
  Returns:
    - generated_data: generated time-series data
    - total_train_time: total training time in seconds
    - infer_time: inference time in seconds
  """
  # Initialization on the Graph
  tf.compat.v1.reset_default_graph()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape

  # If n_generate not specified, generate same number as original
  if n_generate is None:
    n_generate = no

  # Maximum sequence length and each sequence length
  ori_time, max_seq_len = extract_time(ori_data)
  
  def MinMaxScaler(data):
    """Min-Max Normalizer."""
    min_val  = np.min(np.min(data, axis = 0), axis = 0)
    data     = data - min_val
    max_val  = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
    return norm_data, min_val, max_val
  
  # Normalization
  ori_data, min_val, max_val = MinMaxScaler(ori_data)
              
  # Network Parameters
  hidden_dim  = parameters['hidden_dim'] 
  num_layers  = parameters['num_layer']
  iterations  = parameters['iterations']
  batch_size  = parameters['batch_size']
  module_name = parameters['module'] 
  z_dim       = dim
  gamma       = 1
    
  # Input place holders
  X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim],   name = "myinput_x")
  Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name = "myinput_z")
  T = tf.compat.v1.placeholder(tf.int32,   [None],                     name = "myinput_t")
  
  def embedder(X, T):
    with tf.compat.v1.variable_scope("embedder", reuse = tf.compat.v1.AUTO_REUSE):
      e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length = T)
      H = tf.compat.v1.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid)     
    return H
      
  def recovery(H, T):   
    with tf.compat.v1.variable_scope("recovery", reuse = tf.compat.v1.AUTO_REUSE):       
      r_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      r_outputs, r_last_states = tf.compat.v1.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length = T)
      X_tilde = tf.compat.v1.layers.dense(r_outputs, dim, activation=tf.nn.sigmoid) 
    return X_tilde
    
  def generator(Z, T):  
    with tf.compat.v1.variable_scope("generator", reuse = tf.compat.v1.AUTO_REUSE):
      e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length = T)
      E = tf.compat.v1.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid)     
    return E
      
  def supervisor(H, T): 
    with tf.compat.v1.variable_scope("supervisor", reuse = tf.compat.v1.AUTO_REUSE):
      e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers-1)])
      e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length = T)
      S = tf.compat.v1.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid)     
    return S
          
  def discriminator(H, T):
    with tf.compat.v1.variable_scope("discriminator", reuse = tf.compat.v1.AUTO_REUSE):
      d_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      d_outputs, d_last_states = tf.compat.v1.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length = T)
      Y_hat = tf.compat.v1.layers.dense(d_outputs, 1, activation=None) 
    return Y_hat   
    
  # Embedder & Recovery
  H               = embedder(X, T)
  X_tilde         = recovery(H, T)
  E_hat           = generator(Z, T)
  H_hat           = supervisor(E_hat, T)
  H_hat_supervise = supervisor(H, T)
  X_hat           = recovery(H_hat, T)
  Y_fake          = discriminator(H_hat, T)
  Y_real          = discriminator(H, T)     
  Y_fake_e        = discriminator(E_hat, T)
    
  # Variables        
  e_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('embedder')]
  r_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('recovery')]
  g_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('generator')]
  s_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('supervisor')]
  d_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('discriminator')]
    
  # Discriminator loss
  D_loss_real   = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_real),    Y_real)
  D_loss_fake   = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake),   Y_fake)
  D_loss_fake_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
  D_loss        = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
            
  # Generator loss
  G_loss_U   = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake),   Y_fake)
  G_loss_U_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
  G_loss_S   = tf.compat.v1.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,:-1,:])
  G_loss_V1  = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
  G_loss_V2  = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
  G_loss_V   = G_loss_V1 + G_loss_V2
  G_loss     = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V 
            
  # Embedder network loss
  E_loss_T0 = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
  E_loss0   = 10 * tf.sqrt(E_loss_T0)
  E_loss    = E_loss0 + 0.1 * G_loss_S
    
  # Optimizers
  E0_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss0,  var_list = e_vars + r_vars)
  E_solver  = tf.compat.v1.train.AdamOptimizer().minimize(E_loss,   var_list = e_vars + r_vars)
  D_solver  = tf.compat.v1.train.AdamOptimizer().minimize(D_loss,   var_list = d_vars)
  G_solver  = tf.compat.v1.train.AdamOptimizer().minimize(G_loss,   var_list = g_vars + s_vars)      
  GS_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss_S, var_list = g_vars + s_vars)   
        
  sess = tf.compat.v1.Session()
  sess.run(tf.compat.v1.global_variables_initializer())

  # ══════════════════════════════════════════════════════════
  # TRAINING
  # ══════════════════════════════════════════════════════════
  t_train_start = time.time()

  # 1. Embedding network training
  print('Start Embedding Network Training')
  for itt in range(iterations):
    X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)           
    _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})        
    if itt % 1000 == 0:
      print('step: '+ str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)))
  print('Finish Embedding Network Training')

  # 2. Supervised loss only
  print('Start Training with Supervised Loss Only')
  for itt in range(iterations):
    X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)    
    Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
    _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})       
    if itt % 1000 == 0:
      print('step: '+ str(itt) + '/' + str(iterations) + ', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s),4)))
  print('Finish Training with Supervised Loss Only')

  # 3. Joint training
  print('Start Joint Training')
  for itt in range(iterations):
    for kk in range(2):
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)               
      Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
      _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run(
          [G_solver, G_loss_U, G_loss_S, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
      _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})   
    X_mb, T_mb  = batch_generator(ori_data, ori_time, batch_size)           
    Z_mb        = random_generator(batch_size, z_dim, T_mb, max_seq_len)
    check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
    if (check_d_loss > 0.15):        
      _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
    if itt % 1000 == 0:
      print('step: '+ str(itt) + '/' + str(iterations) + 
            ', d_loss: '    + str(np.round(step_d_loss,4)) + 
            ', g_loss_u: '  + str(np.round(step_g_loss_u,4)) + 
            ', g_loss_s: '  + str(np.round(np.sqrt(step_g_loss_s),4)) + 
            ', g_loss_v: '  + str(np.round(step_g_loss_v,4)) + 
            ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0),4)))
  print('Finish Joint Training')

  total_train_time = time.time() - t_train_start
  print(f'\n  ✓ Total training time : {total_train_time:.2f}s')

  # ══════════════════════════════════════════════════════════
  # INFERENCE
  # ══════════════════════════════════════════════════════════
  print(f'\n  Generating {n_generate} synthetic samples …')
  gen_times = [seq_len] * n_generate

  t0 = time.time()
  Z_mb = random_generator(n_generate, z_dim, gen_times, max_seq_len)
  generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, T: gen_times})

  generated_data = []
  for i in range(n_generate):
    temp = generated_data_curr[i, :seq_len, :]
    generated_data.append(temp)

  generated_data = np.array(generated_data) * max_val + min_val
  infer_time = time.time() - t0

  print(f'  ✓ Inference time ({n_generate} samples) : {infer_time:.4f}s')
  print(f'  ✓ Time per sample                      : {infer_time / n_generate * 1000:.4f}ms')

  return generated_data, total_train_time, infer_time