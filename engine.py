import torch
import transformers

from torch import nn
from typing import Callable


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    scores = (predictions * targets).sum(dim=1)
    final_scores = torch.where(scores > 1, 1, torch.where(scores == 1, 0.5, 0))
    return (final_scores.sum() / predictions.shape[0]).item()


def make_pred(pred_emb, ans_embs) -> torch.Tensor:
    cos_sim = nn.CosineSimilarity(dim=2)
    similarities = cos_sim(pred_emb, ans_embs[:len(pred_emb)])
    one_hot_tensor = torch.zeros_like(ans_embs[:len(pred_emb), :, 0])
    max_indices = torch.argmax(similarities, dim=1)
    for dim in range(one_hot_tensor.shape[0]):
        one_hot_tensor[dim, max_indices[dim]] = 1
    return one_hot_tensor


def cos_sim_weighted_loss(pred_emb, ans_embs, weights):
    cos_sim = nn.CosineSimilarity(dim=2)
    similarities = cos_sim(pred_emb, ans_embs[:len(pred_emb)])
    losses = torch.where(weights>0, (1-similarities)*weights, torch.max(torch.zeros(1).to("cuda"), similarities))
    return losses.sum(dim=1).mean()


def train_one_epoch(model: nn.Module,
                    a_model: nn.Module,
                    loader: torch.utils.data.DataLoader, 
                    answers_inp: transformers.tokenization_utils_base.BatchEncoding,
                    optimizer: torch.optim, 
                    a_optimizer: torch.optim,
                    grad_accum_size: int, 
                    acc_fn: Callable[[torch.Tensor, torch.Tensor], float]):
    """
    A function that trains a model by going through all the mini-batches in the training dataloader once.
    """
    print("\tTraining...")
    model.train()
    a_model.train()
    accum_loss = 0  # accumulation of the loss of each batch
    accum_acc = 0  # accumulation of the accuracy of each batch
    
    for i, (X, y, y_acc) in enumerate(loader):
        ans_outputs = a_model(**answers_inp)
        ans_embeddings = ans_outputs[0][:, 0, :].repeat(2, 1, 1)

        outputs = model(**X)
        pred_emb = outputs.pooler_output.unsqueeze(1)
        loss = cos_sim_weighted_loss(pred_emb, ans_embeddings, y)
        loss /= grad_accum_size
        loss.backward()
        pred = make_pred(pred_emb, ans_embeddings)
        acc = acc_fn(pred, y_acc)

        accum_loss += loss.item()
        accum_acc += acc

        if (i + 1) % grad_accum_size == 0:
            optimizer.step()
            optimizer.zero_grad()

            a_optimizer.step()
            a_optimizer.zero_grad()

        print(f"\t\tBatch {i+1}/{len(loader)}: Loss {loss} | Accuracy: {acc*100}%")

    avg_loss = accum_loss / len(loader)
    avg_acc = accum_acc / len(loader)

    return avg_loss, avg_acc


def val_step(model: nn.Module, 
             a_model: nn.Module,
             loader: torch.utils.data.DataLoader, 
             answers_inp: transformers.tokenization_utils_base.BatchEncoding,
             acc_fn: Callable[[torch.Tensor, torch.Tensor], float]):
    """
    A function that validates a model by going through all the mini-batches in the validation dataloader once.
    """
    print("\tValidating...")
    model.eval()
    accum_loss = 0  # accumulation of the loss of each batch
    accum_acc = 0  # accumulation of the accuracy of each batch

    with torch.inference_mode():
        ans_outputs = a_model(**answers_inp)
        ans_embeddings = ans_outputs[0][:, 0, :].repeat(2, 1, 1)

        for i, (X, y, y_acc) in enumerate(loader):
            outputs = model(**X)
            pred_emb = outputs.pooler_output.unsqueeze(1)
            loss = cos_sim_weighted_loss(pred_emb, ans_embeddings, y)
            pred = make_pred(pred_emb, ans_embeddings)
            acc = acc_fn(pred, y_acc)

            accum_loss += loss.item()
            accum_acc += acc

    avg_loss = accum_loss / len(loader)
    avg_acc = accum_acc / len(loader)
    
    return avg_loss, avg_acc


def trainjob(model: nn.Module, 
             a_model: nn.Module,
             epochs: int, 
             train_loader: torch.utils.data.DataLoader, 
             val_loader: torch.utils.data.DataLoader, 
             answers_inp: transformers.tokenization_utils_base.BatchEncoding,
             optimizer: torch.optim, 
             a_optimizer: torch.optim,
             scheduler: torch.optim.lr_scheduler, 
             grad_accum_size: int, 
             acc_fn: Callable[[torch.Tensor, torch.Tensor], float] = accuracy):
    """
    A function to train a model for a specific number of epochs.
    """
    # 4 lists to save the results at the end of each epoch
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        if scheduler is not None:
            print(f"\tLearning rate: {scheduler.get_last_lr()[0]}")
        train_loss, train_acc = train_one_epoch(model, a_model, train_loader, answers_inp, optimizer, a_optimizer, grad_accum_size, acc_fn)
        print(f"\tTrain Loss: {train_loss} | Train Accuracy: {train_acc*100}%")
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = val_step(model, a_model, val_loader, answers_inp, acc_fn)
        print(f"\tValidation Loss: {val_loss} | Validation Accuracy: {val_acc*100}%")
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if scheduler is not None:
            scheduler.step()

    return train_losses, train_accs, val_losses, val_accs
