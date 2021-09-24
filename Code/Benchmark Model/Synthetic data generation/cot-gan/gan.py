import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float32')


class SimpleGenerator(tf.keras.Model):
    '''
    Generator for creating fake time series data (y_1, y_2,...,y_T) from the latent variable Z.
    Args:
         inputs: (numpy array) latent variables as inputs to the RNN model has shape
                 [batch_size, time_step, z_hidden_dims]
    Returns:
          output of RNN generator
    '''
    def __init__(self, batch_size, seq_len, time_steps, sub_seq_len, dx, state_size, z_dims, training_scheme,
                 rnn_activation='sigmoid', output_activation='linear'):
        super(SimpleGenerator, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.time_steps = time_steps
        self.sub_seq_len = sub_seq_len
        self.dx = dx
        dense_layer_unit = max(10, self.sub_seq_len*2)
        self.state_size = state_size
        self.z_dims = z_dims
        self.training_scheme = training_scheme

        self.rnn_activation = rnn_activation
        self.output_activation = output_activation
        self.l2_regularisation = None
        self.counter = 1
        # last lstm output as the input to dense layer
        self.dense_layer = tf.keras.layers.Dense(units=self.state_size, activation='relu', use_bias=True)
        self.dense_layer2 = tf.keras.layers.Dense(units=self.state_size, activation='relu', use_bias=True)
        self.output_layer = tf.keras.layers.Dense(units=self.seq_len, activation=self.output_activation,
                                                  use_bias=True)

    def call(self, inputs, training=True, mask=None):
        y = self.dense_layer(inputs)
        y = self.dense_layer2(y)
        y = self.output_layer(y)
        y = tf.reshape(tensor=y, shape=[self.batch_size, self.seq_len//self.dx, self.dx])
        return y


class ToyGenerator(tf.keras.Model):
    '''
    Generator that combines RNN with FC for creating fake time series data (y_1, y_2,...,y_T)
    from the latent variable Z.
    Args:
        inputs: (numpy array) latent variables as inputs to the RNN model has shape
                [batch_size, time_step, sub_sequence_hidden_dims]
    Returns:
        output of generator
    '''
    def __init__(self, batch_size, time_steps, Dz, Dx, state_size, filter_size, output_activation='sigmoid', bn=False,
                 nlstm=1, nlayer=2, Dy=0, rnn_bn=False):
        super().__init__()

        self.Dz = Dz
        self.Dy = Dy
        self.Dx = Dx
        self.batch_size = batch_size
        self.state_size = state_size
        self.time_steps = time_steps
        self.rnn = tf.keras.Sequential()
        k_init = None
        self.rnn.add(layers.LSTM(self.state_size, return_sequences=True, recurrent_initializer=k_init, kernel_initializer=k_init))
        if rnn_bn:
            self.rnn.add(tf.keras.layers.BatchNormalization())

        for i in range(nlstm-1):
            self.rnn.add(layers.LSTM(self.state_size, return_sequences=True, recurrent_initializer=k_init, kernel_initializer=k_init))
            if rnn_bn:
                self.rnn.add(tf.keras.layers.BatchNormalization())

        self.fc = tf.keras.Sequential()
        for i in range(nlayer-1):
            self.fc.add(layers.Dense(units=filter_size, activation=None, use_bias=True))
            if bn:
                self.fc.add(tf.keras.layers.BatchNormalization())
            self.fc.add(layers.ReLU())
        self.fc.add(layers.Dense(units=Dx, activation=output_activation, use_bias=True))

    def call(self, inputs, y=None, training=True, mask=None):

        z = tf.reshape(tensor=inputs, shape=[self.batch_size, self.time_steps, self.Dz])
        if y is not None:
            y = tf.broadcast_to(y[:, None, :], [self.batch_size, self.time_steps, self.Dy])
            z = tf.concat([z, y], -1)

        lstm = self.rnn(z, training=training)

        x = self.fc(lstm, training=training)
        x = tf.reshape(tensor=x, shape=[self.batch_size, self.time_steps, self.Dx])
        return x


class ToyDiscriminator(tf.keras.Model):
    '''
    1D CNN Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the RNN model has shape [batch_size, time_step, x_dims]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    '''

    def __init__(self, batch_size, time_steps, Dz, Dx, state_size, filter_size, bn=False, kernel_size=5, strides=1,
                 output_activation="tanh", nlayer=2, nlstm=0):
        super().__init__()

        self.batch_size = batch_size
        self.state_size = state_size
        self.time_steps = time_steps
        self.Dz = Dz
        self.Dx = Dx

        self.fc = tf.keras.Sequential()
        self.fc.add(tf.keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size,
                                           padding="causal", strides=strides))

        for i in range(nlayer-1):
            if bn:
                self.fc.add(tf.keras.layers.BatchNormalization())
            self.fc.add(layers.ReLU())
            self.fc.add(tf.keras.layers.Conv1D(filters=state_size, kernel_size=kernel_size,
                                               activation=output_activation if i == nlayer-2 else None,
                                               padding="causal", strides=strides))
        for i in range(nlstm):
            if bn:
                self.fc.add(tf.keras.layers.BatchNormalization())
            self.fc.add(layers.LSTM(state_size, return_sequences=True))

    def call(self, inputs, training=True, mask=None):

        x = tf.reshape(tensor=inputs, shape=[self.batch_size, self.time_steps, self.Dx])
        z = self.fc(x)
        return z


class VideoDCG(tf.keras.Model):
    '''
    Generator for creating fake video sequence (y_1, y_2,...,y_T) from the latent variable Z.
    Args:
         inputs: (numpy array) latent variables as inputs to the RNN layers has shape
                 [batch_size, time_step, z_weight*z_height]
    Returns:
          output of generator: fake video sequence (y_1, y_2,...,y_T)
          of shape [batch_size, x_height, x_weight*time_step, channel]
    '''
    def __init__(self, batch_size, time_steps, x_width, x_height, z_width, z_height, state_size,
                 filter_size=64, bn=False, output_activation="sigmoid", nlstm=1, cat=False, nchannel=3):
        super(VideoDCG, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.x_width = x_width
        self.x_height = x_height
        self.state_size = state_size
        self.z_width = z_width
        self.z_height = z_height
        self.filter_size = filter_size
        self.nlstm = nlstm
        self.cat = cat
        self.nchannel = nchannel

        # last lstm output as the input to dense layer
        self.last_lstm_h = None
        self.bn = bn

        self.lstm_layer1 = tf.keras.layers.LSTM(self.state_size, return_sequences=True)
        if self.bn:
            self.bn1 = tf.keras.layers.BatchNormalization()

        self.lstm_layer2 = tf.keras.layers.LSTM(self.state_size*2, return_sequences=True)

        if self.bn:
            self.bn2 = tf.keras.layers.BatchNormalization()
        
        model = tf.keras.Sequential()
        model.add(layers.Dense(8*8*self.filter_size*4, use_bias=False))
        if self.bn:
            model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((8, 8, self.filter_size*4)))
        # assert model.output_shape == (None, 8, 8, 256) # Note: None is the batch size

        model.add(layers.Conv2DTranspose(self.filter_size * 4, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        # assert model.output_shape == (None, 16, 16, 128)
        if self.bn:
            model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.filter_size*2, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        # assert model.output_shape == (None, 16, 16, 128)
        if self.bn:
            model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        if self.x_width == 64:
            model.add(layers.Conv2DTranspose(self.filter_size, (5, 5), strides=(2, 2), padding='same', use_bias=False))
            # assert model.output_shape == (None, 32, 32, 64)
            if self.bn:
                model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.nchannel, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                         activation=output_activation))

        self.deconv = model

    def call_all(self, inputs_z, inputs_y, training=True, mask=None):
        # for RNN, z has shape of [batch_size, time_step, sub_sequence_hidden_dims]
        z = tf.reshape(tensor=inputs_z, shape=[self.batch_size, self.time_steps, self.z_width*self.z_height])
        y = tf.broadcast_to(inputs_y[:, None, :], [self.batch_size, self.time_steps, inputs_y.shape[-1]])
        zy = tf.concat([z, y], -1)

        lstm_h = self.lstm_layer1(zy)
        if self.cat:
            lstm_h = tf.concat([lstm_h, y], -1)

        if self.bn:
            lstm_h = self.bn1(lstm_h)

        lstm_h = self.lstm_layer2(lstm_h)
        if self.bn:
            lstm_h = self.bn2(lstm_h)

        # input shape for conv3D: (batch, depth, rows, cols, channels)
        conv_inputs = tf.reshape(lstm_h, [self.batch_size * self.time_steps, -1])
        y = self.deconv(conv_inputs)
        
        y = tf.reshape(y, [self.batch_size, self.time_steps, self.x_height, self.x_width, self.nchannel])
        y = tf.transpose(y, (0, 2, 1, 3, 4))
        y = tf.reshape(tensor=y, shape=[self.batch_size, self.x_height, self.x_width*self.time_steps, self.nchannel])
        return zy, lstm_h, y

    def call(self, *args, **kwargs):
        return self.call_all(*args, **kwargs)[-1]


class VideoDCD(tf.keras.Model):
    '''
    Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the model has shape [batch_size, x_height, x_weight*time_step, channel]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    '''

    def __init__(self, batch_size, time_steps, x_width, x_height, z_width, z_height, state_size,
                 filter_size=64, bn=False, nchannel=3):
        super(VideoDCD, self).__init__()

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.x_width = x_width
        self.x_height = x_height
        self.state_size = state_size
        self.z_width = z_width
        self.z_height = z_height
        self.filter_size = filter_size
        self.bn = bn
        self.nchannel = nchannel

        model = tf.keras.Sequential()
        model.add(layers.Conv2D(self.filter_size, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[x_width, x_height, nchannel]))
        if self.bn:
            model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(self.filter_size*2, (5, 5), strides=(2, 2), padding='same'))
        if self.bn:
            model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        if x_width == 64:
            model.add(layers.Conv2D(self.filter_size*4, (5, 5), strides=(2, 2), padding='same'))
            if self.bn:
                model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

        self.conv = model
        
        self.rnn = tf.keras.Sequential()
        if x_width == 64:
            self.rnn.add(tf.keras.layers.LSTM(self.filter_size*4, return_sequences=True))
        elif x_width == 32:
            self.rnn.add(tf.keras.layers.LSTM(self.filter_size*2, return_sequences=True))

        if self.bn:
            self.rnn.add(tf.keras.layers.BatchNormalization())
        self.rnn.add(tf.keras.layers.LSTM(self.state_size, return_sequences=True))

    def call(self, inputs, training=True, mask=None):
        # permute original data shape [batch_size, h, timesteps, w, channels]
        # to [batch_size, timesteps, h, w, channels] as convnet inputs
        z = tf.reshape(tensor=inputs, shape=[self.batch_size, self.x_height, self.time_steps,
                                             self.x_width, self.nchannel])
        z = tf.transpose(z, (0, 2, 1, 3, 4))
        z = tf.reshape(tensor=z, shape=[self.batch_size * self.time_steps, self.x_height, self.x_width, self.nchannel])

        z = self.conv(z)
        z = tf.reshape(z, shape=[self.batch_size, self.time_steps, -1])
        z = self.rnn(z)

        return z