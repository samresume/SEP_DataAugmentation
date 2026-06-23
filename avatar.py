"""AVATAR: Adversarial Autoencoders with Autoregressive Refinement for Time Series Generation."""

from utils import extract_time, random_generator, batch_generator
import time
import numpy as np
import os
import warnings
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def avatar(ori_data, parameters, n_generate=None):
    """AVATAR function.

    Use original data as training set to generate synthetic data (time-series)

    Args:
      - ori_data: original time-series data  (n_samples, seq_len, n_features)
      - parameters: AVATAR network parameters
      - n_generate: number of synthetic samples to generate (default: same as ori_data)

    Returns:
      - generated_data: generated time-series data
      - total_train_time: total training time in seconds
      - infer_time: inference time in seconds
    """
    tf.compat.v1.reset_default_graph()

    no, seq_len, dim = np.asarray(ori_data).shape

    if n_generate is None:
        n_generate = no

    ori_time, max_seq_len = extract_time(ori_data)

    def MinMaxScaler(data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val

    ori_data_norm, min_val, max_val = MinMaxScaler(np.array(ori_data))

    if parameters['hidden_dim'] == 'same':
        hidden_dim = dim
    elif isinstance(parameters['hidden_dim'], int):
        hidden_dim = parameters['hidden_dim']
    else:
        raise ValueError("hidden_dim must be 'same' or a numeric value")

    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    z_dim = hidden_dim

    X = tf.compat.v1.placeholder(
        tf.float32, [None, max_seq_len, dim],   name="myinput_x")
    Z = tf.compat.v1.placeholder(
        tf.float32, [None, max_seq_len, z_dim], name="myinput_z")
    T = tf.compat.v1.placeholder(
        tf.int32,   [None],                     name="myinput_t")

    # ── Network builders ──────────────────────────────────────────
    def make_gru_stack(inputs, T, scope_name, n_layers, hidden_dim, out_dim):
        with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            x = inputs
            for i in range(n_layers):
                with tf.compat.v1.variable_scope(f"layer_{i}"):
                    cell = tf.keras.layers.GRUCell(
                        hidden_dim, activation='tanh')
                    rnn = tf.keras.layers.RNN(cell, return_sequences=True)
                    x = rnn(x)
            out = tf.keras.layers.Dense(out_dim, activation="sigmoid")(x)
        return out

    def make_supervisor_stack(inputs, T, scope_name, n_layers, hidden_dim, out_dim):
        with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            x = inputs
            n = max(1, n_layers - 1)
            for i in range(n):
                with tf.compat.v1.variable_scope(f"layer_{i}"):
                    cell = tf.keras.layers.GRUCell(
                        hidden_dim, activation='tanh')
                    rnn = tf.keras.layers.RNN(cell, return_sequences=True)
                    x = rnn(x)
            out = tf.keras.layers.Dense(out_dim, activation="sigmoid")(x)
        return out

    def discriminator(H, T):
        with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE):
            x = H
            for i in range(num_layers):
                with tf.compat.v1.variable_scope(f"layer_{i}"):
                    cell = tf.keras.layers.GRUCell(
                        hidden_dim, activation='tanh')
                    rnn = tf.keras.layers.RNN(cell, return_sequences=True)
                    x = rnn(x)
            Y_hat = tf.keras.layers.Dense(1, activation=None)(x)
        return Y_hat

    # ── Build graph ───────────────────────────────────────────────
    H = make_gru_stack(X, T, "embedder",    num_layers, hidden_dim, hidden_dim)
    X_tilde = make_gru_stack(H, T, "recovery",    num_layers, hidden_dim, dim)
    X_tilde_supervise = make_supervisor_stack(
        X_tilde, T, "supervisor", num_layers, hidden_dim, dim)
    X_hat_unsupervised = make_gru_stack(
        Z, T, "recovery",    num_layers, hidden_dim, dim)
    X_hat = make_supervisor_stack(
        X_hat_unsupervised, T, "supervisor", num_layers, hidden_dim, dim)
    Y_fake = discriminator(H, T)
    Y_real = discriminator(Z, T)

    e_vars = [v for v in tf.compat.v1.trainable_variables()
              if v.name.startswith('embedder')]
    r_vars = [v for v in tf.compat.v1.trainable_variables()
              if v.name.startswith('recovery')]
    s_vars = [v for v in tf.compat.v1.trainable_variables()
              if v.name.startswith('supervisor')]
    d_vars = [v for v in tf.compat.v1.trainable_variables(
    ) if v.name.startswith('discriminator')]

    # ── Losses ────────────────────────────────────────────────────
    D_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(
        tf.ones_like(Y_real),  Y_real)
    D_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(
        tf.zeros_like(Y_fake), Y_fake)
    D_loss = D_loss_real + D_loss_fake
    R_loss = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
    R_loss_joint = (tf.compat.v1.losses.mean_squared_error(X, X_tilde) +
                    tf.compat.v1.losses.mean_squared_error(X, X_tilde_supervise))
    Ad_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
        tf.ones_like(Y_fake), Y_fake)
    S_loss = (tf.compat.v1.losses.mean_squared_error(X_tilde[:, 1:, :], X_tilde_supervise[:, :-1, :]) +
              tf.compat.v1.losses.mean_squared_error(X_tilde[:, 2:, :], X_tilde_supervise[:, :-2, :]))
    std_loss = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(
        Z, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(H, [0])[1] + 1e-6)))
    mean_loss = tf.reduce_mean(
        tf.abs((tf.nn.moments(Z, [0])[0]) - (tf.nn.moments(H, [0])[0])))
    Distribution_loss = std_loss + mean_loss
    AE_loss = R_loss_joint + Ad_loss + Distribution_loss + S_loss

    # ── Optimizers ────────────────────────────────────────────────
    AE_R_solver = tf.compat.v1.train.AdamOptimizer().minimize(
        R_loss,  var_list=e_vars + r_vars)
    AE_solver = tf.compat.v1.train.AdamOptimizer().minimize(
        AE_loss, var_list=e_vars + r_vars + s_vars)
    D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss,  var_list=d_vars)
    S_solver = tf.compat.v1.train.AdamOptimizer().minimize(
        S_loss,  var_list=r_vars + s_vars)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # ══════════════════════════════════════════════════════════════
    # TRAINING
    # ══════════════════════════════════════════════════════════════
    t_train_start = time.time()

    # 1. Embedding network training
    print('Start Embedding Network Training')
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data_norm, ori_time, batch_size)
        _, step_e_loss = sess.run(
            [AE_R_solver, R_loss], feed_dict={X: X_mb, T: T_mb})
        if itt % 1000 == 0:
            print('step: ' + str(itt) + '/' + str(iterations) +
                  ', e_loss: ' + str(np.round(np.sqrt(step_e_loss), 4)))
    print('Finish Embedding Network Training')

    # 2. Supervised loss only
    print('Start Training with Supervised Loss Only')
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data_norm, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        _, step_g_loss_s = sess.run([S_solver, S_loss], feed_dict={
                                    Z: Z_mb, X: X_mb, T: T_mb})
        if itt % 1000 == 0:
            print('step: ' + str(itt) + '/' + str(iterations) +
                  ', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s), 4)))
    print('Finish Training with Supervised Loss Only')

    # 3. Joint Training
    print('Start Joint Training')
    step_d_loss = 0
    for itt in range(iterations):
        for kk in range(2):
            X_mb, T_mb = batch_generator(ori_data_norm, ori_time, batch_size)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            _, reconstruction, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run(
                [AE_solver, R_loss_joint, Ad_loss, S_loss, Distribution_loss],
                feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
        X_mb, T_mb = batch_generator(ori_data_norm, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        if check_d_loss > 0.15:
            _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={
                                      X: X_mb, T: T_mb, Z: Z_mb})
        if itt % 1000 == 0:
            print('step: ' + str(itt) + '/' + str(iterations) +
                  ', D_loss: ' + str(np.round(step_d_loss, 4)) +
                  ', R_loss_: ' + str(np.round(reconstruction, 4)) +
                  ', Ad_loss_: ' + str(np.round(step_g_loss_u, 4)) +
                  ', S_loss_: ' + str(np.round(np.sqrt(step_g_loss_s), 4)) +
                  ', Distribution_loss: ' + str(np.round(step_g_loss_v, 4)))
    print('Finish Joint Training')

    total_train_time = time.time() - t_train_start
    print(f'\n  ✓ Total training time : {total_train_time:.2f}s')

    # ══════════════════════════════════════════════════════════════
    # INFERENCE
    # ══════════════════════════════════════════════════════════════
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
    print(
        f'  ✓ Time per sample                      : {infer_time / n_generate * 1000:.4f}ms')

    return generated_data, total_train_time, infer_time
