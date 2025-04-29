import os
import torch
import torch.backends.cudnn
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from torch.amp import autocast, GradScaler
from EarlyStop import EarlyStopping
from utils import logger, weight_init
from config import Config
from NewModel import AOD_pono_net  # ✅ 替换为新模型
from data import SmokeDataset
from losses import SSIMLoss


@logger
def load_data(cfg):
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 统一图像大小
        transforms.ToTensor()
    ])
    train_dataset = SmokeDataset(cfg.ori_data_path, cfg.haze_data_path, data_transform)
    val_dataset = SmokeDataset(cfg.test_ori_path, cfg.test_haze_path, data_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size,
                                               shuffle=True, num_workers=cfg.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size,
                                             shuffle=False, num_workers=cfg.num_workers)
    return train_loader, len(train_loader), val_loader, len(val_loader)


@logger
def save_model(epoch, path, net, optimizer, net_name):
    os.makedirs(os.path.join(path, net_name), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(path, net_name, f'AOD_{epoch}.pth'))


@logger
def load_network(device):
    model = AOD_pono_net().to(device)
    model.apply(weight_init)
    return model


@logger
def load_optimizer(model, cfg):
    return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


@logger
def loss_func(device):
    l1 = torch.nn.L1Loss().to(device)
    ssim = SSIMLoss().to(device)

    def combined_loss(pred, target):
        return l1(pred, target) + 0.2 * ssim(pred, target)
    return combined_loss


@logger
def load_summaries(cfg):
    return SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.net_name))


def main(cfg):
    print(cfg)
    if cfg.gpu > -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary = load_summaries(cfg)
    train_loader, train_len, val_loader, val_len = load_data(cfg)
    model = load_network(device)
    criterion = loss_func(device)
    optimizer = load_optimizer(model, cfg)
    scaler = GradScaler("cuda")

    # 初始化EarlyStopping
    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience, save_path=os.path.join(cfg.model_dir, 'best_model.pth'))

    print("Start training")
    model.train()

    for epoch in range(cfg.epochs):
        for step, (ori_img, haze_img) in enumerate(train_loader):
            count = epoch * train_len + (step + 1)
            ori_img, haze_img = ori_img.to(device), haze_img.to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                pred = model(haze_img)
                loss = criterion(pred, ori_img)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            summary.add_scalar('loss', loss.item(), count)
            if step % cfg.print_gap == 0:
                summary.add_image('Input Smoke', make_grid(haze_img[:4], normalize=True), count)
                summary.add_image('Output Clean', make_grid(pred[:4], normalize=True), count)
                summary.add_image('Target Clear', make_grid(ori_img[:4], normalize=True), count)

            print(f"Epoch [{epoch + 1}/{cfg.epochs}] Step [{step + 1}/{train_len}] "
                  f"Loss: {loss.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ------------------- 验证阶段 -------------------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (ori_img, haze_img) in enumerate(val_loader):
                ori_img, haze_img = ori_img.to(device), haze_img.to(device)
                output = model(haze_img)
                loss = criterion(output, ori_img)
                val_loss += loss.item()

        val_loss /= len(val_loader)  # 计算平均验证损失
        print(f"Epoch [{epoch + 1}/{cfg.epochs}] Validation Loss: {val_loss:.6f}")

        # 早停检查
        early_stopping(val_loss, model, optimizer, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break  # 停止训练

        model.train()
        save_model(epoch, cfg.model_dir, model, optimizer, cfg.net_name)

    summary.close()


if __name__ == '__main__':
    cfg = Config()
    main(cfg)
