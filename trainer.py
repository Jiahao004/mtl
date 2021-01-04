import torch
import torch.nn as nn

import os
import logging
import pandas as pd

from tqdm import tqdm
from encoder import MTLEncoderTrainer
from simsiam import SimsiamTrainer


class MTLTrainer:
    def __init__(self,
                 track=pd.DataFrame(columns=['epoch', 'mt_loss', 'mt_ff0', 'mt_ff1', 'simsiam_loss', 'acc0', 'acc1'])):
        self.encoder_trainer = MTLEncoderTrainer(pd.DataFrame(
            columns=['epoch', 'loss', 'p0', 'r0', 'f0', 'pp0', 'rr0', 'ff0', 'p1', 'r1', 'f1', 'pp1', 'rr1', 'ff1']))

        self.simsiam_trainer = SimsiamTrainer(pd.DataFrame(columns=['epoch', 'loss', 'acc0', 'acc1']))

        self.track = track

    def train(self, n_epoch, encoder, simsiam, encoder_optimizer, simsiam_optimizer, train_loader, validation_loader,
              test_loader, device, save_path):
        assert isinstance(encoder, nn.Module)
        best_epoch = 0
        best_acc0 = 0
        for epoch in range(n_epoch):
            mt_loss, simsiam_loss = self.train_epoch(encoder, simsiam, encoder_optimizer, simsiam_optimizer,
                                                     train_loader, device)
            [[p0, r0, f0, pp0, rr0, ff0], acc0] = self.evaluate(encoder, simsiam, validation_loader, device)
            [[p1, r1, f1, pp1, rr1, ff1], acc1] = self.evaluate(encoder, simsiam, test_loader, device)

            if acc0 > best_acc0:
                best_acc0 = acc0
                best_epoch = epoch

            logging.info('''Epoch {:4} Simsiam loss:{:5.4f} val-acc: {:5.4f} test-acc:{:5.4f}
                Multitask Evaluation loss:{.4f}
                    val:    micro: p:{.4f} r:{.4f} f1:{.4f}
                            macro: p:{.4f} r:{.4f} f1:{.4f}
                    test:   micro: p:{.4f} r:{.4f} f1:{.4f}
                            macro: p:{.4f} r:{.4f} f1:{.4f}'''.format(epoch, simsiam_loss, acc0, acc1, mt_loss, p0, r0,
                                                                      f0, pp0, rr0, ff0, p1, r1, f1, pp1, rr1, ff1))

            print('''Epoch {:4} Simsiam loss:{:5.4f} val-acc: {:5.4f} test-acc:{:5.4f}
                Multitask Evaluation loss:{.4f}
                    val:    micro: p:{.4f} r:{.4f} f1:{.4f}
                            macro: p:{.4f} r:{.4f} f1:{.4f}
                    test:   micro: p:{.4f} r:{.4f} f1:{.4f}
                            macro: p:{.4f} r:{.4f} f1:{.4f}'''.format(epoch, simsiam_loss, acc0, acc1, mt_loss, p0, r0,
                                                                      f0, pp0, rr0, ff0, p1, r1, f1, pp1, rr1, ff1))

            self.track.append(pd.DataFrame([epoch, mt_loss, ff0, ff1, simsiam_loss, acc0, acc1],
                                           columns=['epoch', 'mt_loss', 'mt_ff0', 'mt_ff1', 'simsiam_loss', 'acc0',
                                                    'acc1']))
            check_file = {'encoder': encoder.state_dict(),
                          'simsiam': simsiam.state_dict()}
            torch.save(check_file, os.path.join(save_path, 'ep' + str(epoch) + '.cp'))

        logging.info('best epoch is {:4} with acc:{.4f}'.format(best_epoch, best_acc0))
        print('best epoch is {:4} with acc:{.4f}'.format(best_epoch, best_acc0))

    def evaluate(self, encoder, simsiam, test_loader, device, keyword_threshold=0.5, simsiam_threshold=0.5):
        encoder.eval()
        simsiam.eval()
        with torch.no_grad():
            p, r, f, pp, rr, ff = self.encoder_trainer.evaluate_keywords_prediction(encoder, test_loader,
                                                                                    device, keyword_threshold)
            acc = self.simsiam_trainer.evaluate_simsiam(encoder, simsiam, test_loader, device, simsiam_threshold)
        return [[p, r, f, pp, rr, ff], acc]

    def train_epoch(self, encoder, simsiam, encoder_optimizer, simsiam_optimizer, train_loader, device):
        encoder.train()
        simsiam.train()

        mt_loss = 0
        simsiam_loss = 0
        step = 0

        for x, x_tgt, pos, pos_tgt, neg, neg_tgt in tqdm(train_loader):
            l0, l1 = self.train_step(encoder, simsiam, encoder_optimizer, simsiam_optimizer, x, x_tgt, pos, pos_tgt,
                                     neg, neg_tgt, device)
            mt_loss += l0
            simsiam_loss += l1
            step += 1

        return mt_loss / step, simsiam_loss / step

    def train_step(self, encoder, simsiam, encoder_optimizer, simsiam_optimizer, x, x_tgt, pos, pos_tgt, neg, neg_tgt,
                   device):
        mt_loss = self.encoder_trainer.get_loss(encoder, x, x_tgt, device)
        mt_loss += self.encoder_trainer.get_loss(encoder, pos, pos_tgt, device)
        mt_loss += self.encoder_trainer.get_loss(encoder, neg, neg_tgt, device)
        mt_loss.backward()
        encoder_optimizer.step()
        encoder_optimizer.zero_grad()

        simsiam_loss = self.simsiam_trainer.get_loss(encoder, simsiam, x, pos, torch.ones([x.size(0), 1]).to(device),
                                                     device)
        simsiam_loss += self.simsiam_trainer.get_loss(encoder, simsiam, x, neg, torch.zeros([x.size(0), 1]).to(device),
                                                      device)
        simsiam_loss.backward()
        simsiam_optimizer.step()
        simsiam_optimizer.zero_grad()

        return mt_loss.item(), simsiam_loss.item()
