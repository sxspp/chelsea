import torch


class DeepSVDDTrainer:
    
    def __init__(self,normal_list,R):

        self.normal_list = normal_list
        self.R = R

    def __call__(self):
        Dscores = []
        # Dscores와 anchor를 계산
        for i in self.normal_list:
            Dscore = i - self.R ** 2
            Dscores.append(Dscore)

        anchor = torch.stack(self.normal_list, 0)
        anchor = torch.sum(anchor, dim=0)
        anchor /= len(self.normal_list)
        Dscores = torch.stack(Dscores)
        # Dscores와 anchor를 반환
        return Dscores, anchor
