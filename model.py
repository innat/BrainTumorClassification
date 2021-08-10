class BrainTumorModel3D(tf.keras.Model):
    def __init__(self, model, n_gradients=1):
        super(BrainTumorModel3D, self).__init__()
        self.model = model
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), 
                                                  trainable=False) for v in self.model.trainable_variables]
    
    def train_step(self, data):
        self.n_acum_step.assign_add(1)
        
        images, labels = data
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.compiled_loss(labels,
                                      predictions,
                                      regularization_losses=[self.reg_l2_loss()])
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
 
        # If n_acum_step reach the n_gradients then we apply accumulated gradients 
        # to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        predictions = self.model(images, training=False)
        loss = self.compiled_loss(labels, predictions, 
                                  regularization_losses=[self.reg_l2_loss()])
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)
    
    def reg_l2_loss(self, weight_decay = 1e-5):
        return weight_decay * tf.add_n([
            tf.nn.l2_loss(v)
            for v in self.model.trainable_variables
        ])
    
    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, 
                                           self.model.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(
                    self.model.trainable_variables[i], dtype=tf.float32)
            )


from tensorflow.keras import backend as K

class MixDepthGroupConvolution2D(tf.keras.layers.Layer):
    def __init__(self, kernels=[3, 5],
                 conv_kwargs=None,
                 **kwargs):
        super(MixDepthGroupConvolution2D, self).__init__(**kwargs)

        if conv_kwargs is None:
            conv_kwargs = {
                'strides': (1, 1),
                'padding': 'same',
                'dilation_rate': (1, 1),
                'use_bias': False,
            }
        self.channel_axis = -1 
        self.kernels = kernels
        self.groups = len(self.kernels)
        self.strides = conv_kwargs.get('strides', (1, 1))
        self.padding = conv_kwargs.get('padding', 'same')
        self.dilation_rate = conv_kwargs.get('dilation_rate', (1, 1))
        self.use_bias = conv_kwargs.get('use_bias', False)
        self.conv_kwargs = conv_kwargs or {}

        self.layers = [tf.keras.layers.DepthwiseConv2D(kernels[i],
                                       strides=self.strides,
                                       padding=self.padding,
                                       activation=tf.nn.relu,                
                                       dilation_rate=self.dilation_rate,
                                       use_bias=self.use_bias,
                                       kernel_initializer='he_normal')
                        for i in range(self.groups)]

    def call(self, inputs, **kwargs):
        if len(self.layers) == 1:
            return self.layers[0](inputs)
        filters = K.int_shape(inputs)[self.channel_axis]
        splits  = self.split_channels(filters, self.groups)
        x_splits  = tf.split(inputs, splits, self.channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self.layers)]
        return tf.keras.layers.concatenate(x_outputs, axis=self.channel_axis)

    def split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def get_config(self):
        config = {
            'kernels': self.kernels,
            'groups': self.groups,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'conv_kwargs': self.conv_kwargs,
        }
        base_config = super(MixDepthGroupConvolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))