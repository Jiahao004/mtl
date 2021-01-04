import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score


class SimSiam(nn.Module):
    def __init__(self, predictor):
        '''
        :param predictor: usually is a mlp: d_model->d_model
        '''
        super(SimSiam, self).__init__()
        self.predictor = predictor

    def forward(self, encoder, x, y):
        output_x = encoder(x)
        output_y = encoder(y)

        pred_x = self.predictor(output_x)
        pred_y = self.predictor(output_y)

        return output_x, pred_x, output_y, pred_y


class SimsiamTrainer:
    def __init__(self, track):
        self.criterion = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.track = track

    def get_loss(self,encoder, simsiam,x,y,target,device, threshold=0):
        x = x.to(device)
        y = y.to(device)
        target = target.to(device)
        output_x, pred_x, output_y, pred_y = simsiam(encoder, x, y)
        score = ((self.cos(pred_x, output_y.detach()) + self.cos(pred_y, output_x.detach())) / 2).unsqueeze(-1)
        loss = self.criterion(score, target)
        return loss


    def evaluate_simsiam(self, encoder, simsiam, testloader, device, threshold=0.5):
        outputs = []
        targets = []

        for x, x_tgt, pos, pos_tgt, neg, neg_tgt in testloader:
            # x: [b,n_seq,seq]
            x_output = encoder(x)
            x_pred = simsiam(x_output)
            output_pos = encoder(pos)
            cos_score_pos = self.cos(x_pred, output_pos)  # [b]
            output_pos = (cos_score_pos > threshold).to(torch.int).tolist()
            outputs += output_pos
            targets += torch.ones(x.size(0)).tolist()

            output_neg = encoder(neg)
            cos_score_neg = self.cos(x_pred, output_neg)
            output_neg = (cos_score_neg > threshold).to(torch.int).tolist()
            outputs += output_neg
            targets += torch.ones(x.size(0)).tolist()

        acc = accuracy_score(targets, outputs)
        return acc
