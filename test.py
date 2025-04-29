import os
import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image
from torchvision.transforms.functional import resize
from NewModel import AOD_pono_net
from config import Config
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.utils as vutils


def load_model(device, model_path):
    """
    加载训练好的模型
    :param device: 计算设备
    :param model_path: 模型保存路径
    :return: 加载好的模型
    """
    model = AOD_pono_net().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def dehaze_images(cfg):
    """
    对指定文件夹中的图片进行去雾处理，并将去雾前后的图像拼接保存
    :param cfg: 配置信息
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(cfg.model_dir, cfg.net_name, 'best_model.pth')
    model = load_model(device, model_path)

    # 定义图像转换操作
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # 创建输出目录
    os.makedirs(cfg.test_output_dir, exist_ok=True)

    # 遍历测试文件夹中的所有图片
    for filename in os.listdir(cfg.test_haze_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(cfg.test_haze_path, filename)
            # 读取图片
            img = read_image(img_path, mode=ImageReadMode.RGB)
            original_size = img.shape[1:]  # 记录原始图像尺寸

            # 进行图像转换
            input_img = transform(img).unsqueeze(0).to(device)

            # 进行去雾处理
            with torch.no_grad():
                dehazed_img = model(input_img)

            # 将去雾后的图像恢复到原始尺寸
            dehazed_img = resize(dehazed_img.squeeze(0), original_size).unsqueeze(0)

            # 将输入图像也转换为与去雾后图像相同的格式
            input_img = resize(input_img.squeeze(0), original_size).unsqueeze(0)

            # 拼接去雾前后的图像
            combined_img = torch.cat([input_img, dehazed_img], dim=3)

            # 保存拼接后的图像
            output_path = os.path.join(cfg.test_output_dir, filename)
            save_image(combined_img.squeeze(0), output_path)
            print(f"Dehazed {filename} and saved combined image to {output_path}")


if __name__ == '__main__':
    cfg = Config()
    cfg.test_haze_path = "F:/Dehaze/Dehaze/r1/r7"
    cfg.test_output_dir = "test_results"
    dehaze_images(cfg)