import tensorflow as tf
from numpy import genfromtxt
from matplotlib import pyplot as plt
import os
from tqdm import tqdm


class Solver(object):
    def __init__(self, hackoin, log_dir='Logs',
                 model_dir='Model', repeat=10):
        self.hackoin = hackoin
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.repeat = repeat

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

    def load_data(self):
        if self.mode == 'train':
            return genfromtxt('data.csv',
                              delimiter=',')
        elif self.mode == 'eval':
            return genfromtxt('data.csv',
                              delimiter=',')
        else:
            print('incorrect mode')
            exit(1)

    def train(self):
        self.mode = 'train'
        hackoin = self.hackoin
        hackoin.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        checkpoint = tf.train.latest_checkpoint(self.model_dir)

        with tf.Session(config=self.config) as sess:
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # restore variables of Dis and Gen
            try:
                restorer = tf.train.Saver(hackoin.model_params)
                restorer.restore(sess, checkpoint)
                print ('loaded pretrained model')

            except:
                print ('pretrained model not found')

            summary_writer = tf.summary.FileWriter(
                logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver({v.op.name: v for v in hackoin.model_params})

            currency_returns = self.load_data()
            # currency_returns are in reversed order
            currency_returns = currency_returns[::-1]
            train_points, _ = currency_returns.shape

            print ('start training..!')
            for repeat in xrange(self.repeat):
                for step in tqdm(xrange(train_points)):

                    feed_dict = {hackoin.current_currency_returns:
                                 currency_returns[step].reshape((1, 10))
                                 }
                    sess.run(hackoin.update_step, feed_dict=feed_dict)

                    # making summary for analysis
                    summary = sess.run(hackoin.summary_op,
                                       feed_dict)
                    summary_writer.add_summary(summary, repeat * train_points + step)

                    if (step + 1) % 40 == 0:
                        saver.save(sess,
                                   os.path.join(self.model_dir, 'Hackoin'),
                                   global_step=repeat * train_points + step + 1
                                   )

            saver.save(sess,
                       os.path.join(self.model_dir, 'Hackoin'),
                       global_step=repeat * train_points + step + 1)
            coord.request_stop()
            coord.join(threads)

    def eval(self):
        self.mode = 'eval'
        hackoin = self.hackoin
        if not hackoin.built:
            hackoin.build_model()

        checkpoint = tf.train.latest_checkpoint(self.model_dir)

        with tf.Session(config=self.config) as sess:
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # restore variables
            try:
                restorer = tf.train.Saver(hackoin.model_params)
                restorer.restore(sess, checkpoint)
                print ('loaded pretrained model')
            except:
                print 'Could not load pretrained Model'
                exit(1)

            currency_returns = self.load_data()
            # currency_returns are in reversed order
            currency_returns = currency_returns[::-1]
            test_points, _ = currency_returns.shape

            print ('start sampling..!')
            return_summary = []
            volatility_summary = []
            objective_summary = []

            for step in tqdm(xrange(test_points)):

                feed_dict = {
                    hackoin.current_currency_returns: currency_returns[step].reshape((1, 10))}

                returns, volatility, objective = sess.run([hackoin.token_returns,
                                                           hackoin.token_volatility,
                                                           hackoin.obj],
                                                          feed_dict=feed_dict)
                return_summary.append(returns)
                volatility_summary.append(volatility)
                objective_summary.append(objective)

                sess.run(hackoin.update_step, feed_dict=feed_dict)

            coord.request_stop()
            coord.join(threads)

            plt.plot(return_summary)
            plt.xlabel('time')
            plt.ylabel('objective')
            plt.show()
            print objective_summary
