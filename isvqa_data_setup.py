import torch
import torch.nn.functional as F
import json

from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple
from collections import Counter
from utility import ViltSetProcessor, ViTSetProcessor


device = "cuda" if torch.cuda.is_available() else "cpu"


class ISVQA(Dataset):
    """
    A class that loads the ISVQA dataset in the form of a torch.utils.data.Dataset object.
    """
    def __init__(self, qa_path: str, nuscenes_path: str, answers_path: str, processor_type: str, device=device) -> None:
        super().__init__()
        self.device = device
        self.qa_set = self.load_isvqa(qa_path)  # a list of dictionaries that contain all the ImagePaths-Question-Answer pairs
        self.nuscenes_path = nuscenes_path
        self.answers = self.get_isvqa_answers(answers_path)  # a list of all the unique answers in the ISVQA dataset

        if processor_type == "vilt":
            self.processor = ViltSetProcessor(device=device)  # the processor that will take as input the PIL images and the question and will return
                                                          # them in a format compatible to be used as inputs for the MultivieViltModel
        elif processor_type == "vit":
            self.processor = ViTSetProcessor()  # the processor that will take as input the PIL images and the question and will return
                                                # them in a format compatible to be used as inputs for ViT
        else:
            raise ValueError("processor_type should be either 'vilt' or 'vit'.")

    @staticmethod
    def load_isvqa(path: str) -> List[dict]:
        with open(path) as f:
            qa_set = json.load(f)

        return qa_set

    @staticmethod
    def get_isvqa_answers(path: str) -> List[str]:
        with open(path) as f:
            answers = json.load(f)

        return answers 

    def __len__(self) -> int:
        return len(self.qa_set)
    
    def __getitem__(self, index) -> Tuple[dict, torch.Tensor]:
        data = self.qa_set[index]
        question = data["question_str"]
        image_paths = [f"{self.nuscenes_path}/{image_name}.jpg" for image_name in data["image_names"]]

        images = [Image.open(path) for path in image_paths]

        # Get the final inputs that will go into the model
        inputs = self.processor(images, question)

        final_one_hot_answer = torch.zeros(len(self.answers)).to(device)

        for answer in data["answers"]:
            # Turn the answer into an one-hot encoded representation, based on the index of it in the list with all the unique answers
            answer_idx = torch.tensor(self.answers.index(answer))

            one_hot_answer = F.one_hot(answer_idx, len(self.answers)).type(torch.float32).to(self.device)

            final_one_hot_answer += one_hot_answer

        final_one_hot_answer_for_loss = torch.where(final_one_hot_answer > 1, 1, final_one_hot_answer)

        return inputs, final_one_hot_answer_for_loss, final_one_hot_answer
    