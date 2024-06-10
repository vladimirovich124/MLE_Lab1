import torch
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import yaml

from data.download import download_data
from models.cnn import SimpleCNN
from utils.logger import logger

with open('./config/config.yaml') as file:
    config = yaml.safe_load(file)

def train_model(model, trainloader, valloader, config):
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    for epoch in range(config['train']['epochs']):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f'Epoch [{epoch+1}/{config["train"]["epochs"]}], Loss: {running_loss/len(trainloader)}')
        validate_model(model, valloader)

def validate_model(model, valloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logger.info(f'Validation Accuracy: {100 * correct / total}%')

def main():
    trainset, testset = download_data()
    train_data, val_data = train_test_split(trainset, test_size=config['train']['validation_split'], random_state=42)
    trainloader = data.DataLoader(train_data, batch_size=config['train']['batch_size'], shuffle=True)
    valloader = data.DataLoader(val_data, batch_size=config['train']['batch_size'], shuffle=False)
    
    model = SimpleCNN()
    train_model(model, trainloader, valloader, config)

    torch.save(model.state_dict(), config['train']['model_path'])
    logger.info(f'Model saved to {config["train"]["model_path"]}')

if __name__ == "__main__":
    main()