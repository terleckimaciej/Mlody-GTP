# Mlody-GTP
### From Scratch to Pretrained: A Study in Training Small Language Models

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat-square&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=flat-square&logo=huggingface&logoColor=white) ![Google Colab](https://img.shields.io/badge/Google%20Colab-Computational%20Power-F9AB00?style=flat-square&logo=googlecolab&logoColor=white) ![OpenAI](https://img.shields.io/badge/OpenAI-Tokenizer-green?style=flat-square&logo=openai&logoColor=white)

## Introduction
**Mlody-GTP** is a project focused on building a simple Generative Pre-trained Transformer (GPT) from scratch, featuring a manual implementation of the **self-attention** mechanism. The experiment utilizes a self-collected dataset containing the complete discography of Polish rapper **Tede**, concatenated into a single text file.

The primary objective is to investigate whether a model built entirely from the ground up can learn to generate text resembling the Polish language in a song-like manner. Since the model starts with no prior knowledge and operates on individual characters, it must effectively learn how to assemble letters into valid words and syntax. 

Next, we experiment with leveraging different tokenizers by shifting from character-level to BPE-level encoding to understand the challenges that arise regarding data, computation, model size, and pre-training. This provides key insights into how LLMs "learn to speak" and what was required to evolve the self-attention mechanism (introduced in the 2017 [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) paper) into the ChatGPT-like models we take for granted today.

These baseline results are subsequently compared against two evolved approaches:
1.  A model utilizing BPE (Byte Pair Encoding) tokenizer to process sub-word units instead of characters. Multiple applied approaches reveal it's challenges and properties.
2.  A **pre-trained model** where existing weights are recalibrated (fine-tuned) on the same Tede dataset, leveraging Transfer Learning. The idea is to try to overcome dataset and computation limitations of from-scratch-BPE model.

## The Dataset
The models are trained on a corpus of polish rapper **Tede** entire discography lyrics. It consists of ~1M characters.
The data was self acquired via Genius API, one can seamlessly collect its artist of choice eqivalent dataset by changing the ARTIST_NAME parameter and running the [`data_collection.py`](data_collection.py) script. 

## The Three Approaches (Achitecture Evolution)

### Stage 1: From Scratch (Character-Level)
*   **File:** [`gpt.py`](gpt.py)
*   **Architecture:** Custom Transformer built from the ground up using PyTorch.
*   **Tokenization:** Character-level
*   **Concept:** The model starts with zero knowledge of the world or language. It doesn't know what a "word" is. It learns by predicting the next character, based on propability weights learned from the dataset during training.
*   **Observation:** It successfully captures the *visual* structure of lyrics (line breaks, comma placement) and basic phonetic patterns, but may lack semantic understanding, creating non-existing but phonetically reasonable words or even rhymes.

### Stage 2: The Abstraction (BPE Tokenizer)
*   **File:** [`gpt_tiktoken.py`](gpt_tiktoken.py)
*   **Architecture:** Custom Transformer (similar to Stage 1).
*   **Tokenization:** Byte Pair Encoding (BPE). Investigated three distinct approaches:
    1.  **Standard OpenAI Tokenizer:** (50k vocab) 
    2.  **Custom Small-Vocab Tokenizer:** (2k-5k vocab) 
    3.  **Polish-Specific Tokenizer:** (Polish words) 
*   **Concept:** Instead of characters, the model sees "tokens" (chunks of characters, sub-words or whole words). This allows it to process information more efficiently and recognize common word sub-units. We hope to receive only actual words and reasonable clauses or even sentences.
*   **Observation:** This approach revealed a critical "Capacity vs. Density" trade-off.
    *   **Standard BPE (50k):** The model had too many variables (token slots) and not enough data to learn them (a few examples per token).
    *   **Reduced BPE (2k):** Failed due to **Fragmentation**. Tokens were too short (mostly syllables), forcing the model to re-learn spelling from scratch with insufficient data.
    *   **Polish-Specific:** Performed best structurally by treating complex words as single tokens, avoiding spelling errors but still lacking the volume of data to learn flow and rhyme.

### Stage 3: Transfer Learning (Fine-Tuning)
*   **File:** [`gpt_hf.py`](gpt_hf.py)
*   **Architecture:** Pre-trained GPT-2 ([`flax-community/papuGaPT2`](https://huggingface.co/flax-community/papuGaPT2)).
*   **Tokenization:** Pre-trained GPT-2 Tokenizer.
*   **Concept:** I leverage a model that has already "read" massive ammount of Polish language. It understands Polish grammar, declension, and context. It was only fine-tuned to adapt its existing knowledge to the specific *style* of my dataset.
*   **Observation:** 

## Comparative Results

The following snippets demonstrate the evolution of the model's capabilities:

### 1. Character-Level (Visual Imitation)
**based on [`model output`](assets/char_model/output/output1.txt):**

```text
Chłopaki z klałki, taki za mną
Lalk dawno punkt ci, towarli mno
Patrzę nowa w nas weekend wapno
A llub Cicho, liczą się za mną...
```

*   **Phonetic Hallucination:** The model generates non-existent words (*"klałki"*, *"pędziane"*, *"umierka"*, *"Shierzaj"*) that phonetically make sense. It learned *how* Polish sounds without knowing the actual words.
*   **Structurally:** Line lengths and punctuation are mimicked well. It treats text as a visual texture to replicate. Lines even end with rhime-alike.
*   **The "Density" Advantage:** With ~1M characters and only ~100 unique tokens (letters), the model has **~10,000 examples per token**. This massive signal density allows it to master local dependencies (like rhyme suffixes *-no/-mno*) far better than BPE models which struggle with sparsity.
*   **"Sense":** Text doesn't have too much sense... We follow with BPE to try and shift the focus from assembling words to assembling sentences, by taking words "for granted".

### 2. Tiktoken/BPE Evolution (The Data Density Problem)

**a) Low-Vocab (2k) - Syllable Salad:**
> Forced to relearn spelling via short tokens (syllables).
```text
ka chę ć twoje ru chaj ą tam 1 6 00 ! Tak to lecę ogó lnie go zam i mieli s pe ały czas
```

**b) Standard Tiktoken (50k) - Word Salad:**
> Knows words exist but not how to connect them (Sparsity).
```text
W tego, ale dostałem są są leczmy z tobą
Jestem z tamtym sypiączy
Moje pozdrowienia się, życiąż takim klimak, kiedy żyjemy w porzą
```

**c) Polish-Specific BPE - Structural Words:**
> Knows complex words as single tokens, but lacks flow data.
```text
Mam pierwszy patrz, na płyty uwieja w tejjedno baj noga zobowiązani, przekaz
Polski tu po to jak wszystkich śpię
```

### 3. Hugging Face Fine-Tuning 
> "[...]]"

```text
Ej joł, ziom
Jestem Tuzin Gibka co to za mafia ta i inne bauns'y na eBayu!
Buhhh- buuuuhahaha to nie teges z tej strony
pozdrawiam ciebie koleś ze stalowowolskiego osiedla
ten koleżka wiesz tak mnie znają wiem dawno ich poznałem
dziś w sumie chyba nawet oni mi już nic poza tym rapu
nigdy więcej odeszli od mainstreamu...
```

## Usage

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Experiments

**Level 1: Train & Generate (Character-Level)**
```bash
python gpt.py --input assets/input/input2.txt --max_iters 5000
```

**Level 2: Train & Generate (Tiktoken)**
```bash
python gpt_tiktoken.py --input assets/input/input2.txt --max_iters 5000
```

**Level 3: Fine-Tune (Hugging Face)**
```bash
python gpt_hf.py --input assets/input/input2.txt --epochs 3
```
