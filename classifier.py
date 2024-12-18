from collections import Counter
import random
import torch
from tokenizers import Tokenizer
from tqdm import tqdm
import evaluate
import torch.nn.functional as F
import pandas as pd
import csv

# Read and tokenize data
def read_data():
    tokenizer = Tokenizer.from_pretrained("bert-base-cased")

    train = []
    df = pd.read_csv('train.csv')
    text = df['lyrics']
    label = df['genre']
    for x in range(len(text)):
        tokens = tokenizer.encode(text[x]).tokens
        train.append((label[x], tokens))

    dev = []
    df = pd.read_csv('dev.csv')
    text = df['lyrics']
    label = df['genre']
    for x in range(len(text)):
        tokens = tokenizer.encode(text[x]).tokens
        dev.append((label[x], tokens))

    return train, dev

train_data_raw, dev_data_raw = read_data()

# List of all possible genres
df = pd.read_csv('train.csv')
GENRES = sorted(df['genre'].unique())
# Map genres to indices
genre_to_index = {genre: idx for idx, genre in enumerate(GENRES)}

# Define vocabulary class
class Vocab:
    def __init__(self, tokens):
        unique_tokens = list(set(tokens))
        self.vocab = unique_tokens
        self.tok2idx = {tok: idx + 1 for idx, tok in enumerate(self.vocab)}
        self.tok2idx[0] = "[UNK]"
        self.idx2tok = {idx: tok for tok, idx in self.tok2idx.items()}

    def __len__(self):
        return len(self.tok2idx)

    def to_id(self, tok):
        return self.tok2idx.get(tok, 0)

    def to_tok(self, id):
        return self.idx2tok.get(id, "[UNK]")

vocab = Vocab([word for _, tokens in train_data_raw for word in tokens])
VOC_SIZE = len(vocab)

# Process data
def process_data(raw_data):
    data = []
    for label, features in raw_data:
        if label in genre_to_index:
            y = genre_to_index[label]
        else:
            continue  # Skip unknown labels

        x = torch.zeros(VOC_SIZE)
        for feat in features:
            x[vocab.to_id(feat)] += 1

        # Normalize token counts
        x = torch.log1p(x)
        data.append((x, torch.tensor(y)))
    return data

train_data = process_data(train_data_raw)
dev_data = process_data(dev_data_raw)

# Calculate class weights
class_counts = Counter([label for label, _ in train_data_raw])
class_weights = torch.tensor([1.0 / class_counts[genre] for genre in GENRES])

# Define the model
class NNClassifier(torch.nn.Module):
    def __init__(self, voc_size, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(voc_size, num_classes)  # Wx + b

    def forward(self, x):
        y = self.linear(x)
        return y

model = NNClassifier(VOC_SIZE, len(GENRES))
loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Count parameters
def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        print(name, "\t", params)
        total_params += params
    print(f"Total Trainable Params: {total_params}")

count_parameters(model)

# Training loop
def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        xs, ys = zip(*batch)
        yield torch.stack(xs), torch.tensor(ys)

for epoch in range(10):
    print("Epoch", epoch)

    random.shuffle(train_data)
    for x_batch, y_batch in batchify(train_data, batch_size=32):
        model.zero_grad()
        pred = model(x_batch)
        loss = loss_func(pred, y_batch)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        # Evaluate on training data
        total_loss = 0
        for x_batch, y_batch in batchify(train_data, batch_size=32):
            pred = model(x_batch)
            loss = loss_func(pred, y_batch)
            total_loss += loss
        print("train loss:", total_loss / len(train_data))

        # Evaluate on dev data
        total_loss = 0
        for x_batch, y_batch in batchify(dev_data, batch_size=32):
            pred = model(x_batch)
            loss = loss_func(pred, y_batch)
            total_loss += loss
        print("dev loss:", total_loss / len(dev_data))

# Evaluation
def run_model_on_dev_data():
    preds = []
    with torch.no_grad():
        for x, y in dev_data:
            pred = model(x.unsqueeze(0))
            preds.append(pred)
    return preds

def sample_predictions(preds):
    genre_sums = {genre: 0.0 for genre in GENRES}
    for _ in range(10):
        idx = random.randint(0, len(dev_data) - 1)

        # Get the predicted label index
        pred = preds[idx].squeeze(0)  # Remove extra dimensions if any
        pred_idx = torch.argmax(pred).item()
        pred_label = GENRES[pred_idx]

        print("Input:", " ".join(dev_data_raw[idx][1]))
        print("Gold: ", dev_data_raw[idx][0])

        # Ensure probabilities are calculated from a 1D tensor
        probabilities = F.softmax(pred, dim=0)
        print("Pred: ", pred_label)
        print("Probabilities: ", {GENRES[i]: round(probabilities[i].item(), 4) for i in range(len(GENRES))})
        print()

        # Add probabilities to the genre_sums
        for i, genre in enumerate(GENRES):
            genre_sums[genre] += probabilities[i].item()

        # Calculate the average probabilities for each genre
        genre_averages = {genre: round(total / 10, 4) for genre, total in genre_sums.items()}

        # Write the averages to a CSV file
        with open('avgs.csv', mode="w", newline="") as file:
            writer = csv.writer(file)

            # Write header
            writer.writerow(GENRES)

            # Write averages as a single row
            writer.writerow([genre_averages[genre] for genre in GENRES])

preds = run_model_on_dev_data()
sample_predictions(preds)

precision = evaluate.load("precision")
recall = evaluate.load("recall")
accuracy = evaluate.load("accuracy")

refs = [y.item() for _, y in dev_data]
preds_numeric = [torch.argmax(pred).item() for pred in preds]

print("Precision: ", precision.compute(references=refs, predictions=preds_numeric, average="weighted"))
print("Recall: ", recall.compute(references=refs, predictions=preds_numeric, average="weighted"))
print("Accuracy: ", accuracy.compute(references=refs, predictions=preds_numeric))