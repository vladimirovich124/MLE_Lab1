import torch
from data.download import download_data
from models.cnn import SimpleCNN
from utils.logger import logger
import yaml

with open('./config/config.yaml') as file:
    config = yaml.safe_load(file)

def test_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logger.info(f'Test Accuracy: {100 * correct / total}%')

def main():
    _, testset = download_data()
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['train']['batch_size'], shuffle=False)
    
    model = SimpleCNN()
    model.load_state_dict(torch.load(config['train']['model_path']))
    test_model(model, testloader)

if __name__ == "__main__":
    main()