import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
import re

# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Advanced WSD with LSTM/GRU",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Word Sense Disambiguation")
st.caption(
    "Generalized WSD using Bidirectional LSTM/GRU + Target Attention"
)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.sidebar.success(f"Running on: {DEVICE}")


# ============================================================
# 2. DATASET
# ============================================================
#
# Format:
#
# sentence
# target word
# sense ID
# sense description
#
# Add more examples here to improve the model.
# ============================================================

DATA = [

    # ========================================================
    # BANK
    # ========================================================

    ("he deposited money in the bank", "bank", 0,
     "Financial institution"),

    ("she withdrew cash from her bank account", "bank", 0,
     "Financial institution"),

    ("the bank approved my loan", "bank", 0,
     "Financial institution"),

    ("we opened a new bank account", "bank", 0,
     "Financial institution"),

    ("the bank offered me a credit card", "bank", 0,
     "Financial institution"),

    ("the customer deposited money into the bank", "bank", 0,
     "Financial institution"),

    ("the river bank was muddy", "bank", 1,
     "River side or slope"),

    ("we walked along the river bank", "bank", 1,
     "River side or slope"),

    ("the fishermen sat on the grassy bank", "bank", 1,
     "River side or slope"),

    ("the children played near the river bank", "bank", 1,
     "River side or slope"),

    ("the muddy bank was beside the river", "bank", 1,
     "River side or slope"),

    ("trees grew along the river bank", "bank", 1,
     "River side or slope"),


    # ========================================================
    # BARK
    # ========================================================

    ("the dog has a loud bark", "bark", 0,
     "Sound made by a dog"),

    ("the puppy started to bark loudly", "bark", 0,
     "Sound made by a dog"),

    ("the dog's bark woke me up", "bark", 0,
     "Sound made by a dog"),

    ("the dog began to bark at the stranger", "bark", 0,
     "Sound made by a dog"),

    ("the animal barked loudly outside", "bark", 0,
     "Sound made by a dog"),

    ("the bark of the dog was frightening", "bark", 0,
     "Sound made by a dog"),

    ("the bark of the tree was rough", "bark", 1,
     "Outer covering of a tree"),

    ("the tree bark was damaged", "bark", 1,
     "Outer covering of a tree"),

    ("the bark protects the tree", "bark", 1,
     "Outer covering of a tree"),

    ("the rough bark covered the trunk", "bark", 1,
     "Outer covering of a tree"),

    ("the tree had thick brown bark", "bark", 1,
     "Outer covering of a tree"),

    ("she climbed on the bark of the tree", "bark", 1,
     "Outer covering of a tree"),

    ("the bark she climbed on was high", "bark", 1,
     "Outer covering of a tree"),

    ("the children touched the rough tree bark", "bark", 1,
     "Outer covering of a tree"),


    # ========================================================
    # BAT
    # ========================================================

    ("the baseball player swung the bat", "bat", 0,
     "Baseball equipment"),

    ("he bought a wooden bat for baseball", "bat", 0,
     "Baseball equipment"),

    ("the bat hit the baseball", "bat", 0,
     "Baseball equipment"),

    ("the player held the bat tightly", "bat", 0,
     "Baseball equipment"),

    ("a bat flew out of the cave", "bat", 1,
     "Flying mammal"),

    ("the bat was sleeping in the dark cave", "bat", 1,
     "Flying mammal"),

    ("we saw a bat flying at night", "bat", 1,
     "Flying mammal"),

    ("the flying bat entered the cave", "bat", 1,
     "Flying mammal"),


    # ========================================================
    # LIGHT
    # ========================================================

    ("turn on the light in the room", "light", 0,
     "Illumination"),

    ("the light was very bright", "light", 0,
     "Illumination"),

    ("the lamp provides enough light", "light", 0,
     "Illumination"),

    ("the room needs more light", "light", 0,
     "Illumination"),

    ("this bag is very light", "light", 1,
     "Not heavy"),

    ("the box is light enough to carry", "light", 1,
     "Not heavy"),

    ("this material is extremely light", "light", 1,
     "Not heavy"),

    ("the suitcase is surprisingly light", "light", 1,
     "Not heavy"),
]


# ============================================================
# 3. TOKENIZATION
# ============================================================

def tokenize(text):

    text = text.lower()

    # Keep alphabetic words and contractions reasonably clean
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


# ============================================================
# 4. VOCABULARY
# ============================================================

def build_vocab(data, min_frequency=1):

    counter = Counter()

    for sentence, _, _, _ in data:
        counter.update(tokenize(sentence))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, count in counter.items():

        if count >= min_frequency:
            vocab[word] = len(vocab)

    return vocab


vocab = build_vocab(DATA)

PAD_IDX = vocab["<PAD>"]
UNK_IDX = vocab["<UNK>"]

vocab_size = len(vocab)


# ============================================================
# 5. SENSE INFORMATION
# ============================================================

sense_mapping = {}

for sentence, target, sense_id, description in DATA:

    if target not in sense_mapping:
        sense_mapping[target] = {}

    sense_mapping[target][sense_id] = description


target_words = sorted(sense_mapping.keys())


# ============================================================
# 6. DATASET
# ============================================================

class WSDDataset(Dataset):

    def __init__(self, data, vocab):

        self.samples = []

        for sentence, target, label, description in data:

            tokens = tokenize(sentence)

            target = target.lower()

            if target not in tokens:
                continue

            target_position = tokens.index(target)

            ids = [
                vocab.get(word, UNK_IDX)
                for word in tokens
            ]

            self.samples.append({
                "ids": ids,
                "target_position": target_position,
                "label": label,
                "target": target
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


dataset = WSDDataset(DATA, vocab)


# ============================================================
# 7. COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    max_length = max(
        len(item["ids"])
        for item in batch
    )

    inputs = []
    masks = []
    target_positions = []
    labels = []
    targets = []

    for item in batch:

        ids = item["ids"]

        padding = max_length - len(ids)

        padded = ids + [PAD_IDX] * padding

        mask = (
            [True] * len(ids)
            + [False] * padding
        )

        inputs.append(padded)
        masks.append(mask)

        target_positions.append(
            item["target_position"]
        )

        labels.append(item["label"])
        targets.append(item["target"])

    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(masks, dtype=torch.bool),
        torch.tensor(target_positions, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        targets
    )


# ============================================================
# 8. ATTENTION
# ============================================================

class Attention(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, outputs, mask):

        # outputs:
        # [batch, sequence, hidden]

        scores = self.score(outputs).squeeze(-1)

        # Do not attend to padding
        scores = scores.masked_fill(
            ~mask,
            -1e9
        )

        weights = torch.softmax(
            scores,
            dim=1
        )

        context = torch.sum(
            outputs * weights.unsqueeze(-1),
            dim=1
        )

        return context, weights


# ============================================================
# 9. WSD MODEL
# ============================================================

class WSDModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes,
        rnn_type="GRU",
        num_layers=2,
        dropout=0.3
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=PAD_IDX
        )

        if rnn_type == "LSTM":

            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=(
                    dropout
                    if num_layers > 1
                    else 0
                )
            )

        else:

            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=(
                    dropout
                    if num_layers > 1
                    else 0
                )
            )

        # Bi-directional => hidden * 2
        rnn_output_size = hidden_dim * 2

        self.attention = Attention(
            rnn_output_size
        )

        # Target representation + context
        combined_size = (
            rnn_output_size * 2
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                combined_size,
                hidden_dim * 2
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim * 2,
                num_classes
            )
        )

    def forward(
        self,
        x,
        mask,
        target_positions
    ):

        embedded = self.embedding(x)

        outputs, _ = self.rnn(
            embedded
        )

        # ----------------------------------------------------
        # Target-word representation
        # ----------------------------------------------------

        batch_indices = torch.arange(
            outputs.size(0),
            device=outputs.device
        )

        target_vector = outputs[
            batch_indices,
            target_positions
        ]

        # ----------------------------------------------------
        # Attention over sentence
        # ----------------------------------------------------

        context_vector, attention_weights = (
            self.attention(
                outputs,
                mask
            )
        )

        # ----------------------------------------------------
        # Combine target + context
        # ----------------------------------------------------

        combined = torch.cat(
            [
                target_vector,
                context_vector
            ],
            dim=1
        )

        logits = self.classifier(
            combined
        )

        return logits, attention_weights


# ============================================================
# 10. TRAINING
# ============================================================

def train_model(
    model,
    epochs,
    batch_size,
    learning_rate
):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    model.to(DEVICE)

    history = []

    progress = st.progress(0)

    status = st.empty()

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for (
            inputs,
            masks,
            target_positions,
            labels,
            _
        ) in loader:

            inputs = inputs.to(DEVICE)
            masks = masks.to(DEVICE)
            target_positions = (
                target_positions.to(DEVICE)
            )
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits, _ = model(
                inputs,
                masks,
                target_positions
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        average_loss = (
            total_loss / len(loader)
        )

        accuracy = (
            correct / total
        )

        scheduler.step(
            average_loss
        )

        history.append(
            {
                "loss": average_loss,
                "accuracy": accuracy
            }
        )

        progress.progress(
            (epoch + 1) / epochs
        )

        status.write(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    return history


# ============================================================
# 11. SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Model Configuration"
)

rnn_type = st.sidebar.selectbox(
    "Recurrent Model",
    ["GRU", "LSTM"]
)

embedding_dim = st.sidebar.slider(
    "Embedding Dimension",
    16,
    128,
    64,
    16
)

hidden_dim = st.sidebar.slider(
    "Hidden Dimension",
    16,
    128,
    64,
    16
)

num_layers = st.sidebar.slider(
    "Number of RNN Layers",
    1,
    3,
    2
)

dropout = st.sidebar.slider(
    "Dropout",
    0.0,
    0.5,
    0.3,
    0.1
)

epochs = st.sidebar.slider(
    "Training Epochs",
    20,
    200,
    100,
    20
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [4, 8, 16],
    index=1
)

learning_rate = st.sidebar.selectbox(
    "Learning Rate",
    [0.0005, 0.001, 0.003],
    index=1
)


# ============================================================
# 12. CREATE MODEL
# ============================================================

NUM_CLASSES = 2

if (
    "wsd_model" not in st.session_state
    or st.session_state.get("architecture")
    != (
        rnn_type,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout
    )
):

    st.session_state.wsd_model = WSDModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=NUM_CLASSES,
        rnn_type=rnn_type,
        num_layers=num_layers,
        dropout=dropout
    )

    st.session_state.architecture = (
        rnn_type,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout
    )

    st.session_state.trained = False


# ============================================================
# 13. TRAIN BUTTON
# ============================================================

if st.sidebar.button(
    "🚀 Train Model",
    use_container_width=True
):

    st.subheader("📈 Training")

    history = train_model(
        st.session_state.wsd_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    st.session_state.trained = True

    final_loss = history[-1]["loss"]
    final_accuracy = history[-1]["accuracy"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Final Loss",
            f"{final_loss:.4f}"
        )

    with col2:
        st.metric(
            "Training Accuracy",
            f"{final_accuracy * 100:.2f}%"
        )

    st.success(
        f"{rnn_type} training completed."
    )


# ============================================================
# 14. INFERENCE
# ============================================================

def predict(sentence, target_word):

    tokens = tokenize(sentence)

    target_word = target_word.lower()

    if target_word not in tokens:

        return {
            "error":
            f"Target word '{target_word}' "
            "was not found in the sentence."
        }

    target_position = tokens.index(
        target_word
    )

    ids = [
        vocab.get(word, UNK_IDX)
        for word in tokens
    ]

    input_tensor = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )

    mask = torch.tensor(
        [[True] * len(ids)],
        dtype=torch.bool,
        device=DEVICE
    )

    target_position_tensor = torch.tensor(
        [target_position],
        dtype=torch.long,
        device=DEVICE
    )

    model = st.session_state.wsd_model

    model.to(DEVICE)

    model.eval()

    with torch.no_grad():

        logits, attention = model(
            input_tensor,
            mask,
            target_position_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        prediction = probabilities.argmax().item()

    return {
        "tokens": tokens,
        "prediction": prediction,
        "probabilities": probabilities.cpu(),
        "attention": attention[0].cpu()
    }


# ============================================================
# 15. PLAYGROUND
# ============================================================

st.divider()

st.header("🔮 WSD Playground")

target_word = st.selectbox(
    "Choose an ambiguous word",
    target_words
)

examples = {

    "bank":
        "I deposited money into the bank yesterday",

    "bark":
        "the bark she climbed on was too high",

    "bat":
        "the bat flew out of the cave at night",

    "light":
        "this box is very light"
}

sentence = st.text_area(
    "Enter a sentence:",
    value=examples.get(
        target_word,
        f"The {target_word} was interesting."
    ),
    height=100
)


if st.button(
    "🔍 Disambiguate Word",
    use_container_width=True
):

    # Train automatically if necessary
    if not st.session_state.get(
        "trained",
        False
    ):

        with st.spinner(
            "Training model..."
        ):

            train_model(
                st.session_state.wsd_model,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
