from data import LJDataset, collate_fn
from utils import *
import hparams

import torch
from torch.utils.data import DataLoader

import argparse

from datetime import datetime
from tqdm import tqdm

from functional import istft

import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

parser = argparse.ArgumentParser()
parser.add_argument('--torchaudio_path', type=str, default='torchaudio')
parser.add_argument('--data_path', type=str, default='/workspace/raid/data/anagapetyan/DeepGL/data/stc_linear', help='Dataset path')
parser.add_argument('--log_path', type=str, default='/workspace/raid/data/anagapetyan/DeepGL/logdir', help='Path to dir with samples, checkpoints')
parser.add_argument('--pretrained_path',type=str, default='', help='Pretrained model directory')
parser.add_argument('--load_step', type=int, default=0, help='Model load step')
parser.add_argument('--num_blocks', type=int, default=10, help='Number of deep griffin lim blocks')
parser.add_argument('--n_fft', type=int, default=1024, help='FFT window size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--beta1', type=float, default=.9)
parser.add_argument('--beta2', type=float, default=.999)
parser.add_argument('--gamma', type=float, default=.5)
parser.add_argument('--scheduler_step', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--epochs', type=int, default=300, help='Number of epoch for training')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use. >1 uses DataParallel')


class DGLTrainer:
    def __init__(self, args, run_name):
        self.args = args
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = build_model(args)

        if args.pretrained_path:
            self.log_path = args.pretrained_path
        else:
            self.log_path = os.path.join(args.log_path, run_name)

        self.train_dataset = LJDataset(args.data_path, train=True)
        self.test_dataset = LJDataset(args.data_path, train=False)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                      num_workers=args.num_workers, pin_memory=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                     num_workers=args.num_workers, pin_memory=True)
        self.sync_dataloader = DataLoader(self.test_dataset, batch_size=1, collate_fn=collate_fn,
                                     num_workers=args.num_workers, pin_memory=True)

        self.logger = prepare_logger(self.log_path)

    def train(self):
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_step, gamma=self.args.gamma)
        criterion = torch.nn.L1Loss(reduction="sum")        

        min_loss = np.inf
       # self.model.to(self.device)

        for epoch in tqdm(range(self.args.load_step, self.args.epochs)):
            losses = []

            scheduler.step()
            self.model.train()

            for original, noisy in tqdm(self.train_dataloader):
                original, noisy = original.to(self.device), noisy.to(self.device)

                optimizer.zero_grad()

                z, out = self.model(original)
                #print(z.size(), original.size(), out.size())
                loss = criterion(z-original, out)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
               
            self.logger.write_train(np.mean(losses), epoch)

            losses = []

            self.model.eval()

            for original, noisy in self.test_dataloader:
                original, noisy = original.to(self.device), noisy.to(self.device)

                z, out = self.model(noisy)
                loss = criterion(z - original, out)
                losses.append(loss.item())

            self.logger.write_val(np.mean(losses), epoch)

            self.model.eval()

            wavs = []
            for i, (original, noisy) in enumerate(self.sync_dataloader):
                if i < 3:
                    wavs.append(self.gen_wav(noisy.to(self.device)))
                else:
                    break

            self.logger.write_audio(wavs[0], wavs[1], wavs[2], hparams.sample_rate, epoch)

            if np.mean(losses) < min_loss:
                torch.save({
                           'epoch': epoch,
                           'model_state_dict': self.model.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict(),
                           }, os.path.join(os.path.join(self.log_path, 'states'), 'checkpoint{}.pt'.format(epoch)))
                min_loss = np.mean(losses)

    def gen_wav(self, spec):
        self.model.eval()

        z, out = self.model(spec)
        wav = istft(out.permute(0, 2, 3, 1), hparams.fft_size, hop_length=hparams.hop_size)

        return wav.cpu().data.numpy()


if __name__ == '__main__':
    args = parser.parse_args()

    if not args.pretrained_path:
        run_name = datetime.now().strftime('%d:%m:%Y:%H-%M-%S')
    else:
        run_name = ''

    prepare_directories(args, run_name)

    save_parameters(args, run_name)

    trainer = DGLTrainer(args, run_name)

   # prepare_directories(args, run_name)

    trainer.train()

