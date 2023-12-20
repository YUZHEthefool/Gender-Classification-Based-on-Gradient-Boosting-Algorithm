import torch
from homework1.model_cnn2 import Net


def predict_gender(height, weight):
    bmi = weight / ((height * height) / 10000)
    model = Net().cuda()
    model.load_state_dict(torch.load('model_cnn2.pth'))
    model.eval()

    with torch.no_grad():
        inputs = torch.tensor([weight, height], dtype=torch.float).cuda().unsqueeze(0)
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        if 30 >= bmi >= 14:
            if predicted.item() == 1:
                return '男'
            else:
                return '女'
        else:
            return '请输入正确的身高体重'


while True:
    height = float(input("请输入你的身高(厘米): "))
    weight = float(input("请输入你的体重(千克): "))
    print(predict_gender(height, weight))
