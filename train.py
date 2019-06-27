import argparse
import myutils
import torch
from torch import optim, nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import model_builder

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type=str, default='flowers', help='data root')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save the trained model to a checkpoint')
parser.add_argument('--arch', type=str, default='vgg13', help='CNN architecture')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_units', type=int, default=1024, help='hidden units')
parser.add_argument('--epochs', type=int, default=5, help='num of epochs')
parser.add_argument('--gpu', type=str, default='gpu', help='use GPU for training')


in_arg = parser.parse_args()

def train(trainloader, validloader, model, optimizer, criterion, device, epochs, print_every=10):
    to_device = torch.device('cuda' if torch.cuda.is_available() and device=='gpu' else 'cpu')
    steps = 0
    running_loss = 0
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(to_device), labels.to(to_device)
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(to_device), labels.to(to_device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print("Epoch: {}/{}".format(epoch+1, epochs))
                print("Training loss: {:.3f}.. Validation loss: {:.3f}.. Validation accuracy: {:.3f}".format(running_loss/print_every,    valid_loss/len(validloader), accuracy/len(validloader)))
                running_loss = 0
                model.train()
                
def main():
    trainloader, validloader, testloader, train_data = myutils.load_data(in_arg.data_dir)
    model, optimizer = model_builder.create_model(arch=in_arg.arch, hidden_units=in_arg.hidden_units, learnrate=in_arg.learning_rate, device=in_arg.gpu)
    criterion = nn.NLLLoss()
    train(trainloader, validloader, model, optimizer, criterion, in_arg.gpu, in_arg.epochs)
    model.class_to_idx = train_data.class_to_idx
    myutils.save_checkpoint(model, optimizer, path=in_arg.save_dir, arch=in_arg.arch, hidden_units=in_arg.hidden_units, epochs=in_arg.epochs)
    print("====Training completed.====")
    

if __name__ == '__main__':
    main()
    
    
    
    