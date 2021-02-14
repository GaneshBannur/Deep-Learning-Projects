import numpy as np
import random
import torch
from tqdm import tqdm
import torch.nn.functional as F
from contextlib import suppress
from torch.utils.data import DataLoader	


def seed_all(seed=1):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  
def train_test_single_epoch(model, dataloader, loss_function, train, device, optimizer = None):
  if  train and not optimizer:
    raise ValueError("Optimizer required in Train Mode")
  accuracy = 0
  total_loss = 0

  if train:
    model.train()
  else:
    model.eval()
  for (image, label) in tqdm(dataloader, leave=False):
    if train:
      optimizer.zero_grad()

    label = label.to(device)
    with torch.no_grad() if not train else suppress():
      prediction = model(image.to(device))
      loss = loss_function(prediction, label.long())
    if train:
      loss.backward()
      optimizer.step()
      
    total_loss += loss.item()
    probs = F.softmax(prediction)
    accuracy = accuracy + (probs.argmax(dim=1) == label).sum()

  accuracy = accuracy * 100/len(dataloader.dataset)
  avg_loss = total_loss/len(dataloader)
  return accuracy, avg_loss

  
def train_test_classifier(model, train_data, val_data, test_data, batch_size, num_epochs, loss_function, optimizer, logger, device, checkpoint_folder=None, early_stopping_epochs=None):
  train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8)
  val_loader = DataLoader(dataset=val_data, batch_size=batch_size, num_workers=8)
  test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=8)
  val_accuracy_across_epochs = []

  max_accuracy = 0
  max_epoch = 0

  for epoch in range(num_epochs):
    accuracy, avg_loss = train_test_single_epoch(model=model,
                                                 dataloader=train_loader,
                                                 loss_function=loss_function,
                                                 train=True,
                                                 device=device,
                                                 optimizer=optimizer)
    print(f"Train Mode: Epoch {epoch} Accuracy {accuracy: .5f} Loss {avg_loss: .5f}")
    logger.add_scalar("Train/Accuracy", accuracy, epoch)
    logger.add_scalar("Train/Loss", avg_loss, epoch)
    accuracy, avg_loss = train_test_single_epoch(model=model,
                                                 dataloader=val_loader,
                                                 loss_function=loss_function,
                                                 train=False,
                                                 device=device)
    print(f"Val Mode: Epoch {epoch} Accuracy {accuracy: .5f} Loss {avg_loss: .5f}")
    logger.add_scalar("Val/Accuracy", accuracy, epoch)
    logger.add_scalar("Val/Loss", avg_loss, epoch)

    if checkpoint_folder:
      state_to_save = {"model": model.state_dict(),  "optimizer": optimizer.state_dict(), "epoch": epoch}
      torch.save(state_to_save, checkpoint_folder/f"epoch_{epoch}.pt")

    val_accuracy_across_epochs.append(accuracy)
    
    if early_stopping_epochs:
      flag = False
      if len(val_accuracy_across_epochs) >= early_stopping_epochs:
        for a in val_accuracy_across_epochs[-early_stopping_epochs:]:
          if a >= max_accuracy:
            flag = True
        if not flag:
          # reset model to checkpoint with best validation accuracy so far
          checkpoint = torch.load(checkpoint_folder/f"epoch_{max_epoch}.pt")
          model.load_state_dict(checkpoint["model"])
          break
      if val_accuracy_across_epochs[-1] >= max_accuracy:
        max_accuracy = val_accuracy_across_epochs[-1]
        max_epoch = epoch

  accuracy, avg_loss = train_test_single_epoch(model=model,
                                               dataloader=test_loader,
                                               loss_function=loss_function,
                                               train=False,
                                               device=device)
  print(f"Test Mode: Accuracy {accuracy: .5f} Loss {avg_loss: .5f}") 