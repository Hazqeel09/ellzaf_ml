import torch

def save_checkpoint(epoch, model, model_name, max_accuracy, optimizer, lr_scheduler):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,}

    save_path = f'weights/{model_name}_ckpt_epoch_{epoch}.pth'
    print(f"{save_path} saving......")
    torch.save(save_state, save_path)
    print(f"{save_path} saved !!!")