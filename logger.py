import numpy as np
from tensorboardX import SummaryWriter


class Logger(SummaryWriter):
    def __init__(self, logdir):
        self.logger_dir = logdir
        super(Logger, self).__init__(self.logger_dir)

    def write_train(self, training_loss, iteration):
        self.add_scalar('training loss', training_loss, iteration)

    def write_val(self, validation_loss, iteration):
        self.add_scalar('validation loss', validation_loss, iteration)

    def write_audio(self, audio1, audio2, audio3, sample_rate, iteration):
        self.add_audio('sample', audio1, iteration, sample_rate)
        self.add_audio('sample', audio2, iteration, sample_rate)
        self.add_audio('sample', audio3, iteration, sample_rate)

        np.save(self.logger_dir + '/sample{}'.format(iteration), audio1)
        np.save(self.logger_dir + '/sample{}'.format(iteration), audio2)
        np.save(self.logger_dir + '/sample{}'.format(iteration), audio3)
