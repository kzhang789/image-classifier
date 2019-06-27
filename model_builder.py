import torch
from torch import optim, nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
    def forward(self, x):
        for fc in self.hidden_layers:
            x = F.relu(fc(x))
            x = self.dropout(x)
        x = F.log_softmax(self.output(x), dim=1)
        return x
    
def create_model(arch='vgg13', hidden_units=1024, learnrate = 0.001, device='gpu'):
    to_device = torch.device('cuda' if torch.cuda.is_available() and device=='gpu' else 'cpu')
    
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_size = 9216
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        in_size = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_size = 1024
    else:
        print("Not one of 'vgg13', 'densenet121' or 'alexnet'.")
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = Classifier(in_size, 102, [hidden_units])

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)

    model.to(to_device)
    return model, optimizer