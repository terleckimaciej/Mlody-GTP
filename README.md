# Mlody-GTP: Evolution of Generative AI Architectures

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat-square&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=flat-square&logo=huggingface&logoColor=white) ![Google Colab](https://img.shields.io/badge/Google%20Colab-Computational%20Power-F9AB00?style=flat-square&logo=googlecolab&logoColor=white) ![OpenAI](https://img.shields.io/badge/OpenAI-Tiktoken-green?style=flat-square&logo=openai&logoColor=white)

## Introduction
**Mlody-GTP** is a project focused on building a simple Generative Pre-trained Transformer (GPT) from scratch, featuring a manual implementation of the **self-attention** mechanism. The experiment utilizes a self-collected dataset containing the complete discography of Polish rapper **Tede**, concatenated into a single text file.

The primary objective is to investigate whether a model built entirely from the ground up, using **character-level tokenization**, can learn to generate text resembling the Polish language in a song-like manner. Since the model starts with no prior knowledge and operates on individual characters, it must effectively learn how to assemble letters into valid words and syntax.

These baseline results are subsequently compared against two evolved approaches:
1.  A model utilizing OpenAI's **tiktoken** (BPE) tokenizer to process sub-word units instead of characters.
2.  A **pre-trained model** where existing weights are recalibrated (fine-tuned) on the same Tede dataset, leveraging Transfer Learning.

## The Dataset
The models are trained on a corpus of polish rapper **Tede** entire discography lyrics. This dataset presents unique challenges:
*   **Unstructured Nature:** Unlike prose, lyrics rely on rhythm, line breaks, and flow.
*   **Slang & Neologisms:** High frequency of non-standard vocabulary not found in formal dictionaries.
*   **Complex Rhyme Schemes:** Patterns that require anticipating sounds lines ahead.
The file was self acquired via Genius API, one can seamlessly collect its artist of choice eqivalent dataset by changing the ARTIST_NAME parameter and running the `data_collection.py` script.

## The Three Approaches (Achitecture Evolution)

### Stage 1: From Scratch (Character-Level)
*   **File:** `gpt.py`
*   **Architecture:** Custom Transformer built from the ground up using PyTorch.
*   **Tokenization:** Character-level: (example text)->(e,x,a,m,p,l,e, ,t,e,x,t)->(1,2,3,4,5,6,1,8,9,1,2,9)
*   **Concept:** The model starts with zero knowledge of the world or language. It doesn't know what a "word" is. It learns by predicting the next character, based on propability weights learned from the dataset during training
*   **Observation:** It successfully captures the *visual* structure of lyrics (line breaks, comma placement) and some basic phonetic patterns, but may lack semantic understanding.

### Stage 2: The Abstraction (BPE Tokenizer)
*   **File:** `gpt_tiktoken.py`
*   **Architecture:** Custom Transformer (similar to Stage 1).
*   **Tokenization:** Byte Pair Encoding (BPE) using [`OpenAI's tiktoken`](https://tiktokenizer.vercel.app): (example text)->(example, ⋅text)->(18582, 2201)
*   **Concept:** Instead of characters, the model sees "tokens" (chunks of characters or whole words). This allows it to process information more efficiently and recognize common word sub-units.
*   **Observation:** The output may contain real Polish words, but because the model is trained from scratch on a small dataset, it lacks the volume of data required to learn grammar from zero. It resembles "word salad"—correct vocabulary, semi-random usage.
> Using same-sized dataset as for char-level model may raise concerns, because BPE encoding causes massive raise in token dictionary and was done just for the experiment needs. Normally we would need many times larger dataset. 

### Stage 3: Transfer Learning (Fine-Tuning)
*   **File:** `gpt_hf.py`
*   **Architecture:** Pre-trained GPT-2 ([`flax-community/papuGaPT2`](https://huggingface.co/flax-community/papuGaPT2)).
*   **Tokenization:** Pre-trained GPT-2 Tokenizer.
*   **Concept:** I leverage a model that has already "read" massive ammount of Polish language. It understands Polish grammar, declension, and context. It was only fine-tuned to adapt its existing knowledge to the specific *style* of my dataset.
*   **Observation:** Full stylistic adoption. The model generates coherent sentences that rhyme, use slang correctly, and maintain a consistent theme.

## Comparative Results

The following snippets demonstrate the evolution of the model's capabilities:

### 1. Character-Level (Visual Imitation)
> "Visual imitation only. Looks like lyrics from far away, gibberish up close."

```text
rustmachi ryjnych kim tu hipisan! / To lecie miałem o sobie
```

### 2. Tiktoken/BPE (Word Salad)
> "Real words, no thoughts. Random association of concepts."

```text
Retro Mercedes Benz - #Hot Vinci / Represent, Kikt się miał...
```

### 3. Hugging Face Fine-Tuning (Coherence)
> "Full stylistic adoption. Coherent sentences."

```text
Ej joł ziomek! Co się u nas dzieje? Od rana do wieczora...
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
