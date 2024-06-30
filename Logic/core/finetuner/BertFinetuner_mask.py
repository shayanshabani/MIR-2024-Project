from huggingface_hub import login, logout

import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score , classification_report

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizer,AutoModelForSequenceClassification, get_linear_schedule_with_warmup,TrainingArguments,Trainer

class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        # TODO: Implement initialization logic

        self.model = None

        self.top_n_genres = top_n_genres
        self.file_path = file_path

        self.dataset = []
        self.training_set = None
        self.validation_set = None
        self.test_set = None

        self.threshold = 0.5

        self.genre_dict = {}
        self.genre_list_label = []
        self.genre_dict_label = {}
        self.genre_frequency = {}

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        # TODO: Implement dataset loading logic
        with open(self.file_path, 'r') as file:
            self.dataset = json.load(file)

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        # TODO: Implement genre filtering and visualization logic
        another = []
        for movie in self.dataset:
            if movie['genres'] and movie['first_page_summary'] and len(movie['genres']) > 0 and movie['first_page_summary'] != '':
                another.append(movie)

        self.dataset = another

        self.genre_frequency = {genre: self.genre_frequency.get(genre, 0) + 1 for movie in self.dataset for genre in movie['genres']}

        print(self.genre_frequency)

        top_genres = sorted(self.genre_frequency, key=self.genre_frequency.get, reverse=True)[:self.top_n_genres]

        self.genre_dict = {genre: i for i, genre in enumerate(top_genres)}
        self.genre_dict_label = {str(i): genre for genre, i in self.genre_dict.items()}
        self.genre_list_label = list(self.genre_dict.keys())

        labels_matrix = np.zeros((len(self.dataset), self.top_n_genres))
        for i, movie in enumerate(self.dataset):
            genres = movie['genres']
            if genres:
                for genre in genres:
                    if genre in self.genre_dict:
                        labels_matrix[i][self.genre_dict[genre]] = 1
            movie['genres'] = labels_matrix[i]


    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        # TODO: Implement dataset splitting logic
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=42)
        self.training_set = train
        self.validation_set, self.test_set = train_test_split(test, test_size=0.5, random_state=42)

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        # TODO: Implement dataset creation logic
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, epochs=1, batch_size=128, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        # TODO: Implement BERT fine-tuning logic

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            problem_type="multi_label_classification",
            id2label=self.genre_dict_label,
            label2id=self.genre_dict,
            num_labels=self.top_n_genres
        )

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device).train()

        training_token = self.tokenizer(
            [movie['first_page_summary'] for movie in tqdm(self.training_set)],
            truncation=True, padding=True
        )

        validation_token = self.tokenizer(
            [movie['first_page_summary'] for movie in tqdm(self.validation_set)],
            truncation=True, padding=True
        )

        train = self.create_dataset(training_token, [movie['genres'] for movie in self.training_set])

        validation = self.create_dataset(validation_token, [movie['genres'] for movie in self.validation_set])

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=".",
                evaluation_strategy="epoch",
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs
            ),
            train_dataset=train,
            eval_dataset=validation,
            optimizers=(
                torch.optim.AdamW(self.model.parameters(), lr=5e-5, eps=1e-8, weight_decay=weight_decay),
                get_linear_schedule_with_warmup(
                    torch.optim.AdamW(self.model.parameters(), lr=5e-5, eps=1e-8, weight_decay=weight_decay),
                    num_warmup_steps=warmup_steps,
                    num_training_steps=len(self.training_set) * epochs)
            )
        ).train()


    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        # TODO: Implement metric computation logic
        predictions, actual_labels = pred
        activation_function = torch.nn.Sigmoid()
        probabilities = activation_function(torch.Tensor(predictions))
        predicted_labels = np.zeros(probabilities.shape)
        predicted_labels[probabilities >= self.threshold] = 1
        ground_truth = actual_labels
        result = classification_report(ground_truth, predicted_labels, target_names=self.genre_dict_label)
        return result

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        # TODO: Implement model evaluation logic

        test_token = self.tokenizer(
            [movie['first_page_summary'] for movie in tqdm(self.test_set)],
            truncation=True, padding=True
        )

        test = self.create_dataset(test_token, [movie['genres'] for movie in tqdm(self.test_set)])

        test_loader = torch.utils.data.DataLoader(test, batch_size=16, shuffle=False)

        self.model.eval()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        predictions, actual_labels = [], []
        for batch in test_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels']
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(logits.tolist())
            actual_labels.extend(labels.tolist())

        print(self.compute_metrics((predictions, [movie['genres'] for movie in self.test_set])))


    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        # TODO: Implement model saving logic
        self.model.save_pretrained("Bert/" + model_name)
        self.tokenizer.save_pretrained("Bert/" + model_name)
        self.model.push_to_hub(model_name)
        self.tokenizer.push_to_hub(model_name)

class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        # TODO: Implement initialization logic
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        # TODO: Implement item retrieval logic
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        # TODO: Implement length computation logic
        return len(self.labels)

