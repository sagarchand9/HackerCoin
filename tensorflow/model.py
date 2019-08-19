import tensorflow as tf
import tensorflow.contrib.slim as slim


class Hackoin(object):
    def __init__(self, learning_rate=1e-3, alpha=0.8,
                 tau=1.0, layers=[10, 15, 15], n_currencies=10):
        self.tau = tau
        self.alpha = alpha
        self.layers = layers
        self.n_currencies = n_currencies
        self.learning_rate = learning_rate

        # self.current_weights = tf.Variable()  # todo: initialize properly

        self.token_weights = []
        self.token_prices = []
        self.token_returns = []
        self.token_volatility = []

        self.ema = 0     # exponential moving average
        self.emv = 0     # exponential moving variance

        self.built = False

        self.Weights = []
        self.biases = []

    def generate_weights(self):
        with tf.variable_scope('weight_predictor'):
            self.Weights.append(tf.Variable(
                tf.random_normal((self.n_currencies, self.layers[0]),
                                 mean=1e-3)))
            self.biases.append(tf.Variable(tf.random_normal(
                (self.layers[0],), mean=1e-3)))

            for l in xrange(1, len(self.layers)):
                W = tf.Variable(tf.random_normal((self.layers[l - 1],
                                                  self.layers[l]),
                                                 mean=1e-3))
                b = tf.Variable(tf.random_normal((self.layers[l],),
                                                 mean=1e-3))
                self.Weights.append(W)
                self.biases.append(b)

            W = tf.Variable(tf.random_normal((self.layers[- 1],
                                              self.n_currencies),
                                             mean=1e-3))
            b = tf.Variable(tf.random_normal((self.n_currencies,),
                                             mean=1e-3))
            self.Weights.append(W)
            self.biases.append(b)
            self.model_params = self.Weights + self.biases

    def compute_weights(self):
        net = self.current_currency_returns

        for W, b in zip(self.Weights, self.biases):
            net = tf.nn.relu(tf.add(tf.matmul(net, W), b))
        net = tf.nn.softmax(net)

        return net

    def build_model(self):
        self.generate_weights()
        self.current_currency_returns = tf.placeholder(tf.float32,
                                                       [1, self.n_currencies])
        self.current_weights = self.compute_weights()
        current_return = tf.reduce_mean(tf.matmul(
            self.current_currency_returns,
            tf.transpose(self.current_weights)))
<<<<<<< HEAD
        print("sagarchand") 
        print()
        if self.ema is None:
            self.ema = current_return
        else:
            self.ema += self.alpha * (current_return - self.ema)
=======

        self.ema += (1 - self.alpha) * (current_return - self.ema)
>>>>>>> 9cb34737da1d897b03b0d1db134c691ee865aa27

        current_volatility = (1 - self.alpha) * (self.emv + self.alpha *
                                                 tf.square(current_return - self.ema))
        self.emv = current_volatility

        self.token_weights.append(self.current_weights)
        # self.token_prices.append(current_price)
        self.token_returns.append(current_return)
        self.token_volatility.append(current_volatility)

        # ---------------------------------- #
        #        OBJECTIVE FUNCTION          #
        # ---------------------------------- #

        self.obj = current_volatility - self.tau * self.ema

        self.update_step = tf.train.RMSPropOptimizer(
            self.learning_rate).minimize(self.obj)

        # ---------------------------------- #
        #           SUMMARY OP               #
        # ---------------------------------- #

        token_weight_summary = tf.summary.histogram('token_weight',
                                                    self.current_weights)
        # token_price_summary = tf.summary.scalar('token_price',
        #                                         current_price)
        token_return_summary = tf.summary.scalar('token_return',
                                                 current_return)
        objective_summary = tf.summary.scalar('objective',
                                              self.obj)
        token_volatility_summary = tf.summary.scalar('token_volatility',
                                                     current_volatility)

        self.summary_op = tf.summary.merge([
            token_weight_summary,
            objective_summary,
            # token_price_summary,
            token_return_summary,
            token_volatility_summary])
        self.built = True
