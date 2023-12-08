import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn

class RDFTrainer():
    def __init__(self,RDF_list,anchor):

        self.RDF_list = RDF_list
        self.anchor = anchor

    def __call__(self):

        CS = nn.CosineSimilarity(dim = 0)

        positive = []
        negative = []
        addition = []
        for i in range(len(self.RDF_list)):
            
            result = CS(torch.tensor(self.RDF_list[i]).clone().detach(), torch.tensor(self.anchor).clone().detach())

            if result >=0.2:
                positive.append(self.RDF_list[i])
            else:
                negative.append(self.RDF_list[i])

        if len(positive) != 0 and len(negative) != 0:

            if len(positive) > len(negative):

                addition = torch.zeros([int(len(positive)-len(negative)),128]).cuda()

                for i in addition:

                    negative.append(i)
                
            elif len(positive) < len(negative):

                addition = torch.zeros([int(len(negative)-len(positive)),128]).cuda()
                for i in addition:

                    positive.append(i)

            positive_tensor = torch.stack(positive)
            negative_tensor = torch.stack(negative)

            
        elif len(positive) != 0 and len(negative) == 0:

            positive_tensor = torch.stack(positive)
            negative_tensor = torch.zeros([len(self.RDF_list),128]).cuda()

        elif len(positive) == 0 and len(negative) != 0:
            
            negative_tensor = torch.stack(negative)
            positive_tensor = torch.zeros([len(self.RDF_list),128]).cuda()

            


        pos_dist = (self.anchor - positive_tensor)**2
        neg_dist = (self.anchor - negative_tensor)**2


        return pos_dist, neg_dist


    
