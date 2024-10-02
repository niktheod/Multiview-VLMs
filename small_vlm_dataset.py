import torch
import torch.nn.functional as F
import json

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Tuple


device = "cuda" if torch.cuda.is_available() else "cpu"

class SmallVLMDataset(Dataset):
    def __init__(self, qa_path: str, nuscenes_path: str, answers_path: str, vocab, device=device) -> None:
        super().__init__()
        self.device = device
        self.qa_set = self.load_isvqa(qa_path)  # a list of dictionaries that contain all the ImagePaths-Question-Answer pairs
        self.nuscenes_path = nuscenes_path
        self.answers = self.get_isvqa_answers(answers_path)  # a list of all the unique answers in the ISVQA dataset
        self.vocab = vocab
        self.max_size = 40

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

        question_tokens = data["question_tokens"]
        token_ids = torch.zeros(self.max_size, dtype=torch.int64).to(self.device)
        for i, token in enumerate(question_tokens):
            if token in self.vocab:
                token_ids[i] = self.vocab.index(token) + 1
            else:
                token_ids[i] = 0
        
        length = torch.tensor([len(data["question_tokens"])], dtype=torch.int64)

        image_paths = [f"{self.nuscenes_path}/{image_name}.jpg" for image_name in data["image_names"]]

        images = [Image.open(path) for path in image_paths]

        mean = [0.3776, 0.3823, 0.3740]
        std = [0.1904, 0.1872, 0.1929]

        transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
        
        tensor_images = [transform(image) for image in images]

        tensor_images = torch.stack(tensor_images).to(self.device)
        final_one_hot_answer = torch.zeros(len(self.answers)).to(self.device)

        for answer in data["answers"]:
            # Turn the answer into an one-hot encoded representation, based on the index of it in the list with all the unique answers
            answer_idx = torch.tensor(self.answers.index(answer))

            one_hot_answer = F.one_hot(answer_idx, len(self.answers)).type(torch.float32).to(self.device)

            final_one_hot_answer += one_hot_answer

        final_one_hot_answer_for_loss = torch.where(final_one_hot_answer > 1, 1, final_one_hot_answer)

        return token_ids, length, tensor_images, final_one_hot_answer_for_loss, final_one_hot_answer
    