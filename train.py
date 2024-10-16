import torch
import os
import json
import matplotlib.pyplot as plt
import datetime

from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from isvqa_data_setup import ISVQA
from nuscenesqa_data_setup import NuScenesQA
from nuscenesmqa_data_setup import NuScenesMQA
from models import MultiviewViltForQuestionAnsweringBaseline, DoubleVilt, ImageSetQuestionAttention
from engine import trainjob
from nuscenesmqa_engine import nuscenesmqa_trainjob
from nuscenes.nuscenes import NuScenes
from typing import List, Tuple
from prettytable import PrettyTable


device = "cuda" if torch.cuda.is_available() else "cpu"


def save_plots(results: Tuple[List[float], List[float], List[float], List[float]], path: str):
    """
    A function to save plots of the results after training is done.
    """
    plt.plot(range(1, len(results[0])+1), results[0], label="Training")
    plt.plot(range(1, len(results[2])+1), results[2], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{path}/loss.png", facecolor="white")

    plt.clf()

    plt.plot(range(1, len(results[1])+1), ([x*100 for x in results[1]]), label="Training")
    plt.plot(range(1, len(results[3])+1), ([x*100 for x in results[3]]), label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")
    plt.legend()
    plt.savefig(f"{path}/accuracy.png", facecolor="white")

    plt.clf()

    plt.plot(range(1, len(results[0])+1), results[0], label="Training")
    plt.plot(range(1, len(results[2])+1), results[2], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.yscale('log')
    plt.legend()
    plt.savefig(f"{path}/loss_log.png", facecolor="white")

    plt.clf()


def count_parameters(model: nn.Module):
    """
    A function that prints the trainble parameters of the model that will be trained.
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    cnt = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        cnt += 1
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params



def train(hyperparameters: defaultdict,
          model_variation: str,
          dataset: str,
          qa_path: str,
          nuscenes_path: str,
          path_to_save_results: str,
          path_to_save_model: str,
          results_subfolder: str = None,
          pretrained_baseline: bool = True,
          fine_tune_all: bool = False,
          img_lvl_pos_emb : bool = False,
          best_baseline: str = None,
          scheduler_type: str = None,
          device: str = device):
    """
    A function that automatically runs the training pipeline.
    First, it creates the dataset and dataloader objects that will be used for training and validation.
    Second, it initializes the model that will be trained and changes the head of it (classifier) to match the number of answers of the dataset of choice.
    Then, it runs the training job according to the defined hyperpaameters.
    Finally, it saves the results (numbers and plots) as wel as a setup file with all the details (hyperparameters, dataset etc.) of the current training job.
    """
    
    torch.set_float32_matmul_precision("high")

    seed = hyperparameters["seed"] if hyperparameters["seed"] is not None else 42

    generator = torch.Generator().manual_seed(seed)  # set a generator for reproducable results
    train_percentage = hyperparameters["train_percentage"]
    val_percentage = hyperparameters["val_percentage"]
    affix = ".csv" if dataset == "nuscenesmqa" else ".json"

    # Define the paths for the train, val, and test sets
    if train_percentage == "trainval":
        train_path = f"{qa_path}/trainval_set{affix}"
        val_path = f"{qa_path}/test_set{affix}"
    else:
        if train_percentage is None or train_percentage == 100:
            train_path = f"{qa_path}/train_set{affix}"
        else:
            train_path = f"{qa_path}/train_set_{train_percentage}{affix}"
        if val_percentage is None or val_percentage == 100:
            val_path = f"{qa_path}/val_set{affix}"
        else:
            val_path = f"{qa_path}/val_set_{val_percentage}{affix}"
    answers_path = f"{qa_path}/answers.json"

    processor_type = "vit" if model_variation == "vit_vilt" else "vilt"
    
    # Load the dataset (either ISVQA or NuScenesQA)
    if dataset == "isvqa":
        train_set = ISVQA(qa_path=train_path,
                        nuscenes_path=nuscenes_path,
                        answers_path=answers_path,
                        processor_type=processor_type,
                        device=device)
        
        val_set = ISVQA(qa_path=val_path,
                        nuscenes_path=nuscenes_path,
                        answers_path=answers_path,
                        processor_type=processor_type,
                        device=device)

        num_answers = len(train_set.answers)
    elif dataset == "nuscenesqa":
        dataroot = nuscenes_path[:-8]
        nusc = NuScenes(version="v1.0-trainval", dataroot=dataroot, verbose=False)

        train_set = NuScenesQA(qa_path=train_path,
                               nusc=nusc,
                               nuscenes_path=nuscenes_path,
                               answers_path=answers_path,
                               processor_type=processor_type,
                               device=device)
        
        val_set = NuScenesQA(qa_path=val_path,
                             nusc=nusc,
                             nuscenes_path=nuscenes_path,
                             answers_path=answers_path,
                             processor_type=processor_type,
                             device=device)

        num_answers = len(train_set.answers)
    elif dataset == "nuscenesmqa":
        dataroot = nuscenes_path[:-8]
        nusc = NuScenes(version="v1.0-trainval", dataroot=dataroot, verbose=False)

        train_set = NuScenesMQA(qa_path=train_path,
                               nusc=nusc,
                               nuscenes_path=nuscenes_path,
                               answers_path=answers_path,
                               processor_type=processor_type,
                               device=device)
        
        val_set = NuScenesMQA(qa_path=val_path,
                             nusc=nusc,
                             nuscenes_path=nuscenes_path,
                             answers_path=answers_path,
                             processor_type=processor_type,
                             device=device)

        num_answers = len(train_set.answers)

    batch_size = hyperparameters["batch_size"]

    # Define the train and val dataloaders
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              generator=generator)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False)
    
    # Define some values necessary for the initialization of the model
    set_size = hyperparameters["set_size"] if hyperparameters["set_size"] is not None else 6
    img_seq_len = hyperparameters["img_seq_len"] if hyperparameters["img_seq_len"] is not None else 210
    question_seq_len = hyperparameters["question_seq_len"] if hyperparameters["question_seq_len"] is not None else 40
    emb_dim = hyperparameters["emb_dim"] if hyperparameters["emb_dim"] is not None else 768

    if emb_dim != 768 and pretrained_baseline == True:
        raise ValueError("For pretrained ViLT the only valid value of emd_dim is 768")

    # Define the model
    if model_variation == "baseline":
        blind  = hyperparameters["blind"] if hyperparameters["blind"] is not None else False
        model = MultiviewViltForQuestionAnsweringBaseline(set_size, img_seq_len, emb_dim, pretrained_baseline, pretrained_baseline, img_lvl_pos_emb, blind=blind).to(device)

        # If we use pretrained weights and we don't want to fine tune the whole model (we only want to learn the VQA head), then we set requires_grad = False for all the other parameters.
        if not fine_tune_all and pretrained_baseline:
            for param in model.parameters():
                param.requires_grad = False

        model.model.classifier = nn.Sequential(
            nn.Linear(emb_dim, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Linear(1536, num_answers)
        ).to(device)
    elif model_variation == "double_vilt":
        pretrained_final_model = hyperparameters["double_vilt"]["pretrained_final_model"]

        model = DoubleVilt(set_size, img_seq_len, question_seq_len, emb_dim, pretrained_baseline, pretrained_final_model,
                                                  pretrained_model_path=best_baseline).to(device)

        train_baseline, train_final_model = hyperparameters["double_vilt"]["train_baseline"], hyperparameters["double_vilt"]["train_final_model"]

        if train_baseline and train_final_model:
            pass
        else:
            for name, parameter in model.named_parameters():
                if not train_baseline and not train_final_model:
                    if name[:8] != "img_attn":
                        parameter.requires_grad = False
                elif not train_baseline:
                    if name[:11] != "final_model" and name[:8] != "img_attn":
                        parameter.requires_grad = False
                elif not train_final_model:
                    if name[:8] != "baseline" and name[:8] != "img_attn":
                        parameter.requires_grad = False
                

        model.final_model.classifier = nn.Sequential(
            nn.Linear(emb_dim, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Linear(1536, num_answers)
        ).to(device)
    elif model_variation == "vit_vilt":
        model = ImageSetQuestionAttention(**hyperparameters["vit_vilt"]).to(device)
        
        if not hyperparameters["vit_vilt"]["train_vilt"]:
            for name, parameter in model.named_parameters():
                if name[:4] != "attn":
                    parameter.requires_grad = False

        model.vilt.classifier = nn.Sequential(
            nn.Linear(emb_dim, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Linear(1536, num_answers)
        ).to(device)
    else:
        raise ValueError("model_variation should be either 'baseline' or 'double_vilt' or 'vit_vilt'")
    
    # Print the parameters to be trained
    print("Parameters to be trained: ")
    count_parameters(model)

    # Define and setup all the necessary hyperparameters and objects for training
    weight_decay = hyperparameters["weight_decay"] if hyperparameters["weight_decay"] is not None else 0

    optimizer_name = "adam" if hyperparameters["optimizer_name"] is None else hyperparameters["optimizer_name"]

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"], weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters["lr"], weight_decay=weight_decay)

    if scheduler_type == "steplr":
        step_size = hyperparameters["scheduler_step_size"]
        gamma = hyperparameters["scheduler_gamma"]
        if step_size is None:
            raise ValueError("hyperparameters['scheduler_step_size'] should be defined.")
        if gamma is None:
            raise ValueError("hyperparameters['scheduler_gamma'] should be defined.")
        scheduler = StepLR(optimizer=optimizer,
                           step_size=step_size,
                           gamma=gamma)
    elif scheduler_type is not None:
        raise ValueError("scheduler_type should be either 'steplr' or 'None'")
    else:
        scheduler = None

    epochs = hyperparameters["epochs"]

    grad_accum_size = hyperparameters["grad_accum_size"] if hyperparameters["grad_accum_size"] is not None else 1

    comp_model = torch.compile(model)

    # Run the training job
    if dataset == "nuscenesmqa":
        results = nuscenesmqa_trainjob(comp_model, epochs, train_loader, val_loader, optimizer, scheduler, grad_accum_size)
    else:
        results = trainjob(comp_model, epochs, train_loader, val_loader, optimizer, scheduler, grad_accum_size)

    # Define a setup dictionary that will be saved together with the results, in order to be able to remeber what setup gave the corresponding results
    if model_variation == "baseline":
        setup = {"model_variation": model_variation,
                    "dataset": dataset,
                    "pretrained_baseline": pretrained_baseline,
                    "fine_tune_all": fine_tune_all,
                    "img_lvl_pos_emb": img_lvl_pos_emb,
                    "seed": seed,
                    "train_percentage": train_percentage,
                    "val_percentage": val_percentage,
                    "emb_dim": emb_dim,
                    "epochs": epochs,
                    "optimizer": optimizer_name,
                    "lr": hyperparameters["lr"],
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "grad_accum_size": grad_accum_size,
                    "scheduler": scheduler_type,
                    "scheduler_step_size": hyperparameters["scheduler_step_size"],
                    "scheduler_gamma": hyperparameters["scheduler_gamma"],
                    "blind": blind
                    }
    elif model_variation == "double_vilt":
        setup = {"model_variation": model_variation,
                    "dataset": dataset,
                    "pretrained_baseline": pretrained_baseline,
                    "img_lvl_pos_emb": img_lvl_pos_emb,
                    "seed": seed,
                    "train_percentage": train_percentage,
                    "val_percentage": val_percentage,
                    "emb_dim": emb_dim,
                    "epochs": epochs,
                    "optimizer": optimizer_name,
                    "lr": hyperparameters["lr"],
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "grad_accum_size": grad_accum_size,
                    "best_baseline": best_baseline,
                    "scheduler": scheduler_type,
                    "scheduler_step_size": hyperparameters["scheduler_step_size"],
                    "scheduler_gamma": hyperparameters["scheduler_gamma"],
                    "double_vilt": hyperparameters["double_vilt"]
                    }
    else:
        setup = {"model_variation": model_variation,
                    "dataset": dataset,
                    "img_lvl_pos_emb": img_lvl_pos_emb,
                    "seed": seed,
                    "train_percentage": train_percentage,
                    "val_percentage": val_percentage,
                    "emb_dim": emb_dim,
                    "epochs": epochs,
                    "optimizer": optimizer_name,
                    "lr": hyperparameters["lr"],
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "grad_accum_size": grad_accum_size,
                    "scheduler": scheduler_type,
                    "scheduler_step_size": hyperparameters["scheduler_step_size"],
                    "scheduler_gamma": hyperparameters["scheduler_gamma"],
                    "vit_vilt": hyperparameters["vit_vilt"]
                    }

    # Save the model and the results
    title = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results_folder = f"{path_to_save_results}/{results_subfolder}/{title}"
    model_folder = f"{path_to_save_model}/{title}"

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(model.state_dict(), f"{model_folder}/model.pth")
    with open(f"{model_folder}/setup.json", "w") as f:
        json.dump(setup, f)

    with open(f"{results_folder}/results.json", "w") as f:
        json.dump(results, f)

    with open(f"{results_folder}/setup.json", "w") as f:
        json.dump(setup, f)

    save_plots(results=results,
               path=results_folder)
    
    print("Done!")
    print(f"Results filename: {title}")
