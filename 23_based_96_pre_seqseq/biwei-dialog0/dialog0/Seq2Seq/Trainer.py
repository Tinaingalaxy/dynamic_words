from __future__ import division
import time
import os
import sys
import math

def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans
class Statistics(object):
    """
    Train/validate loss statistics.
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def ppl(self):
        return safe_exp(self.loss / self.n_words)

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def elapsed_time(self):
        return time.time() - self.start_time

    def print_out(self, epoch, batch, n_batches):
        t = self.elapsed_time()

        out_info = ("Epoch %2d, %5d/%5d| acc: %6.2f| ppl: %6.2f| " + \
               "%3.0f tgt tok/s| %4.0f s elapsed") % \
              (epoch, batch, n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_words / (t + 1e-5),
               time.time() - self.start_time)

        print(out_info)
        sys.stdout.flush()

class Trainer(object):
    def __init__(self, opt, model, train_iter,
                 train_loss, optim, lr_scheduler):

        self.model = model
        self.train_iter = train_iter
        self.train_loss = train_loss
        self.optim = optim
        self.lr_scheduler = lr_scheduler

        # Set model in training mode.
        self.model.train()

        self.global_step = 0
        self.step_epoch = 0
        self.opt = opt

    def update(self, batch, shard_size):
        self.model.zero_grad()
        src_inputs = batch.src[0]
        src_lengths = batch.src[1].tolist()
        tgt_inputs = batch.tgt[0][:-1]
        tag_inputs = batch.tag
        # print("tag_inputs,主要查看有多少个:",tag_inputs)
        outputs = self.model((src_inputs,tgt_inputs,tag_inputs,src_lengths))
        stats = self.train_loss.sharded_compute_loss(batch, outputs, shard_size)

        self.optim.step()
        return stats

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics()
        report_stats = Statistics()

        for step_batch, batch in enumerate(self.train_iter):
            self.global_step += 1

            stats = self.update(batch, 32)

            report_stats.update(stats)
            total_stats.update(stats)

            if report_func is not None:
                report_stats = report_func(self.global_step,
                        epoch, step_batch, len(self.train_iter),
                        self.optim.lr, report_stats)


        return total_stats

    def save_per_epoch(self, epoch, out_dir):
        f = open(os.path.join(out_dir,'checkpoint'),'w')
        f.write('latest_checkpoint:checkpoint_epoch%d.pkl'%(epoch))
        f.close()
        self.model.drop_checkpoint(epoch,self.opt,
                    os.path.join(out_dir,"checkpoint_epoch%d.pkl"%(epoch)))

    def load_checkpoint(self, cpnt):
        self.model.load_checkpoint(cpnt)