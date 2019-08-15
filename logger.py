import numpy as np
from tensorboardX import SummaryWriter
import os

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

        np.save(os.path.join(self.logger_dir, 'samples') + '/sample1_{}'.format(iteration), audio1)
        np.save(os.path.join(self.logger_dir, 'samples') + '/sample2_{}'.format(iteration), audio2)
        np.save(os.path.join(self.logger_dir, 'samples') + '/sample3_{}'.format(iteration), audio3)


if __name__ == '__main__':
     from datetime import datetime
     import numpy as np
     run_name = datetime.now().strftime('%d:%m:%Y:%H-%M-%S')
     log_dir = '/workspace/raid/data/anagapetyan/DeepGL/logdir/'+run_name
     logger = Logger(log_dir)
     os.mkdir(os.path.join(log_dir, 'samples'))
     a = np.random.rand(100)
     logger.write_audio(a,a,a,16000,0)

     print('check saved audio in', '/workspace/raid/data/anagapetyan/DeepGL/logdir/'+run_name)
