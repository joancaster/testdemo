import torch.nn.functional as F
from torch.nn import init
#the git is working
#?

class hswish(nn.Module):
    # H -swish activation funtions
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

model_weight_path = ".pth"
pre_weights = torch.load(model_weight_path, map_location=device)
pre_dict = {k:v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
net.load_state_dict(pre_dict, strict=False)

class hsigmoid(nn.Module):
    # hsigmoid activation functions
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out
    def init_params(self):
    #Initialization parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
class SeModule(nn.Module):
    #SE attintion block
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
