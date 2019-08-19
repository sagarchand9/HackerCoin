import tensorflow as tf
from model import Hackoin
from solver import Solver


def get_flags():
    flags = tf.app.flags
    flags.DEFINE_string('mode', 'train', "'train' or 'test'")
    flags.DEFINE_float('learning_rate', 1e-9, "learning rate for RMSProp")

    FLAGS = flags.FLAGS
    return FLAGS


def main(_):
    hackoin = Hackoin(learning_rate=1e-9, alpha=0.3,
                      tau=4e-6, layers=[10], n_currencies=10)
    model_dir = './Model/'
    log_dir = './Logs/'
    solver = Solver(hackoin, model_dir=model_dir, log_dir=log_dir, repeat=120)

    if FLAGS.mode == 'train':
        if not tf.gfile.Exists(model_dir):
            tf.gfile.MakeDirs(model_dir)
        solver.train()
    else:
        solver.eval()


if __name__ == '__main__':
    FLAGS = get_flags()
    tf.app.run(main=main)
