import streamlit as st
import re
import math
from collections import Counter, defaultdict

# Optional imports
try:
    import nltk
    from nltk import word_tokenize, pos_tag
    from nltk.corpus import brown
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="N-Gram & POS Tagging",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 N-Gram Model & POS Tagging")
st.write(
    "Enter text and select a POS-tagging model: "
    "Rule-Based, Probabilistic, HMM, or Transformer."
)


# ---------------------------------------------------------
# DOWNLOAD NLTK DATA
# ---------------------------------------------------------
if NLTK_AVAILABLE:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger")
        except:
            pass

    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown")


# ---------------------------------------------------------
# SAMPLE TEXT
# ---------------------------------------------------------
default_text = (
    "The quick brown fox jumps over the lazy dog. "
    "The dog watches the fox."
)

text = st.text_area(
    "Enter your text:",
    value=default_text,
    height=150
)


# ---------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------
model = st.selectbox(
    "Select POS Tagging Model",
    [
        "Rule-Based",
        "Probabilistic",
        "HMM",
        "Transformer"
    ]
)


# ---------------------------------------------------------
# N-GRAM SELECTION
# ---------------------------------------------------------
st.subheader("N-Gram Model")

ngram_type = st.selectbox(
    "Select N-Gram",
    [
        "Unigram",
        "Bigram",
        "Trigram"
    ]
)


# ---------------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------------
def simple_tokenize(text):
    return re.findall(r"\b\w+\b|[.!?,;:]", text)


tokens = simple_tokenize(text)


# ---------------------------------------------------------
# RULE-BASED POS TAGGER
# ---------------------------------------------------------
def rule_based_tagger(tokens):
    tagged = []

    determiners = {
        "a", "an", "the", "this", "that",
        "these", "those", "my", "your",
        "his", "her", "our", "their"
    }

    pronouns = {
        "i", "you", "he", "she", "it",
        "we", "they", "me", "him",
        "her", "us", "them"
    }

    prepositions = {
        "in", "on", "at", "by", "for",
        "with", "from", "to", "of",
        "over", "under", "into"
    }

    conjunctions = {
        "and", "or", "but", "because",
        "although", "while", "if"
    }

    auxiliaries = {
        "is", "am", "are", "was", "were",
        "be", "been", "being", "has",
        "have", "had", "do", "does", "did"
    }

    for word in tokens:
        lower = word.lower()

        if lower in determiners:
            tag = "DT"

        elif lower in pronouns:
            tag = "PRP"

        elif lower in prepositions:
            tag = "IN"

        elif lower in conjunctions:
            tag = "CC"

        elif lower in auxiliaries:
            tag = "VB"

        elif word in ".,!?;:":
            tag = "."

        elif lower.endswith("ing"):
            tag = "VBG"

        elif lower.endswith("ed"):
            tag = "VBD"

        elif lower.endswith("ly"):
            tag = "RB"

        elif lower.endswith("ous") or lower.endswith("ful"):
            tag = "JJ"

        elif lower.endswith("ness"):
            tag = "NN"

        elif lower.endswith("s"):
            tag = "NNS"

        elif lower.isdigit():
            tag = "CD"

        else:
            tag = "NN"

        tagged.append((word, tag))

    return tagged


# ---------------------------------------------------------
# PROBABILISTIC TAGGER
# ---------------------------------------------------------
def probabilistic_tagger(tokens):
    if not NLTK_AVAILABLE:
        return rule_based_tagger(tokens)

    try:
        return pos_tag(tokens)
    except Exception:
        return rule_based_tagger(tokens)


# ---------------------------------------------------------
# HMM TAGGER
# ---------------------------------------------------------
class HMMTagger:

    def __init__(self):
        self.transition = defaultdict(Counter)
        self.emission = defaultdict(Counter)
        self.tag_counts = Counter()
        self.tags = set()

    def train(self, sentences):

        for sentence in sentences:

            previous_tag = "<START>"

            for word, tag in sentence:

                word = word.lower()

                self.transition[previous_tag][tag] += 1
                self.emission[tag][word] += 1

                self.tag_counts[tag] += 1
                self.tags.add(tag)

                previous_tag = tag

            self.transition[previous_tag]["<END>"] += 1

    def transition_prob(self, previous, current):

        total = sum(self.transition[previous].values())

        if total == 0:
            return 1e-6

        return (
            (self.transition[previous][current] + 1)
            / (total + len(self.tags) + 1)
        )

    def emission_prob(self, tag, word):

        total = self.tag_counts[tag]

        if total == 0:
            return 1e-6

        return (
            (self.emission[tag][word.lower()] + 1)
            / (total + len(self.emission[tag]) + 1)
        )

    def tag(self, words):

        if not words:
            return []

        # Viterbi
        V = [{}]
        backpointer = [{}]

        for tag in self.tags:

            V[0][tag] = (
                math.log(self.transition_prob("<START>", tag))
                + math.log(self.emission_prob(tag, words[0]))
            )

            backpointer[0][tag] = None

        for i in range(1, len(words)):

            V.append({})
            backpointer.append({})

            for current_tag in self.tags:

                best_previous = None
                best_score = float("-inf")

                emission = math.log(
                    self.emission_prob(current_tag, words[i])
                )

                for previous_tag in self.tags:

                    score = (
                        V[i - 1][previous_tag]
                        + math.log(
                            self.transition_prob(
                                previous_tag,
                                current_tag
                            )
                        )
                        + emission
                    )

                    if score > best_score:
                        best_score = score
                        best_previous = previous_tag

                V[i][current_tag] = best_score
                backpointer[i][current_tag] = best_previous

        best_last = max(
            V[-1],
            key=V[-1].get
        )

        best_path = [best_last]

        for i in range(len(words) - 1, 0, -1):
            best_last = backpointer[i][best_last]
            best_path.append(best_last)

        best_path.reverse()

        return list(zip(words, best_path))


@st.cache_resource
def train_hmm():

    if not NLTK_AVAILABLE:
        return None

    try:
        sentences = brown.tagged_sents()

        hmm = HMMTagger()
        hmm.train(sentences)

        return hmm

    except Exception:
        return None


# ---------------------------------------------------------
# TRANSFORMER TAGGER
# ---------------------------------------------------------
@st.cache_resource
def load_transformer():

    if not TRANSFORMERS_AVAILABLE:
        return None

    try:
        return pipeline(
            "token-classification",
            model="vblagoje/bert-english-uncased-finetuned-pos"
        )
    except Exception:
        return None


def transformer_tagger(text):

    tagger = load_transformer()

    if tagger is None:
        return None

    try:
        result = tagger(text)

        tagged = []

        for item in result:

            word = item["word"]
            tag = item["entity"]

            # Remove WordPiece markers
            word = word.replace("##", "")

            tagged.append((word, tag))

        return tagged

    except Exception:
        return None


# ---------------------------------------------------------
# N-GRAM MODEL
# ---------------------------------------------------------
class NGramModel:

    def __init__(self, n=2):

        self.n = n
        self.ngrams = Counter()
        self.contexts = Counter()

    def train(self, tokens):

        tokens = [
            "<START>"
        ] * (self.n - 1) + tokens + ["<END>"]

        for i in range(len(tokens) - self.n + 1):

            gram = tuple(
                tokens[i:i + self.n]
            )

            context = gram[:-1]

            self.ngrams[gram] += 1
            self.contexts[context] += 1

    def probability(self, gram):

        context = gram[:-1]

        return (
            self.ngrams[gram] + 1
        ) / (
            self.contexts[context] + len(self.ngrams) + 1
        )

    def predict_next(self, context):

        candidates = []

        context = tuple(context[-(self.n - 1):])

        for gram in self.ngrams:

            if gram[:-1] == context:

                prob = self.probability(gram)

                candidates.append(
                    (gram[-1], prob)
                )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return candidates[:10]


# ---------------------------------------------------------
# DISPLAY N-GRAM RESULTS
# ---------------------------------------------------------
def show_ngram():

    n = {
        "Unigram": 1,
        "Bigram": 2,
        "Trigram": 3
    }[ngram_type]

    model = NGramModel(n)
    model.train([x.lower() for x in tokens])

    st.write(f"### {ngram_type} Frequencies")

    rows = []

    for gram, count in model.ngrams.items():

        if "<START>" not in gram and "<END>" not in gram:

            rows.append({
                "N-Gram": " ".join(gram),
                "Count": count,
                "Probability": round(
                    model.probability(gram),
                    4
                )
            })

    rows = sorted(
        rows,
        key=lambda x: x["Count"],
        reverse=True
    )

    st.dataframe(
        rows,
        use_container_width=True
    )

    # Prediction
    if n > 1 and len(tokens) >= n - 1:

        context = tokens[-(n - 1):]

        st.write(
            f"### Next-word prediction for: "
            f"`{' '.join(context)}`"
        )

        predictions = model.predict_next(context)

        if predictions:

            for word, probability in predictions:
                st.write(
                    f"- **{word}** → "
                    f"{probability:.4f}"
                )
        else:
            st.info(
                "No matching next-word prediction "
                "was found in the input."
            )


# ---------------------------------------------------------
# POS TAGGING
# ---------------------------------------------------------
def perform_pos_tagging():

    st.subheader("POS Tagging Result")

    if model == "Rule-Based":

        tagged = rule_based_tagger(tokens)

    elif model == "Probabilistic":

        tagged = probabilistic_tagger(tokens)

    elif model == "HMM":

        hmm = train_hmm()

        if hmm is None:
            st.error(
                "HMM model could not be loaded."
            )
            return

        tagged = hmm.tag(tokens)

    else:

        tagged = transformer_tagger(text)

        if tagged is None:

            st.error(
                "Transformer model could not be loaded. "
                "Install the required packages."
            )

            return

    # Table
    data = []

    for word, tag in tagged:

        data.append({
            "Word": word,
            "POS Tag": tag
        })

    st.dataframe(
        data,
        use_container_width=True
    )

    # Inline result
    st.write("### Tagged Sentence")

    output = " ".join(
        f"{word}/{tag}"
        for word, tag in tagged
    )

    st.code(output)


# ---------------------------------------------------------
# BUTTON
# ---------------------------------------------------------
if st.button(
    "🚀 Perform POS Tagging & N-Gram Analysis",
    type="primary"
):

    if not text.strip():

        st.warning(
            "Please enter some text."
        )

    else:

        tab1, tab2 = st.tabs(
            [
                "🏷️ POS Tagging",
                "🔤 N-Gram Analysis"
            ]
        )

        with tab1:
            perform_pos_tagging()

        with tab2:
            show_ngram()
