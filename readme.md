# CompeteModel : Models used in the project [MindScape India (Compete)](https://github.com/riziuzi/compete)
This repository contains all the models currently under development and does not handle the hosting of these models. (refer to the [Compete](https://github.com/riziuzi/compete) repository)
## Setup



System Setup [Video link](https://www.youtube.com/watch?v=VE5OiQSfPLg)
<details>
<summary>Documented steps</summary>

```html
Tenserflow GPU(2.14) installation on Windows 11 through WSL2 ( VS Code installation and Jupiter LAB installation included)
1.GPU Drivers update
2.Create Windows Subsystem for Linux (WSL)
 2.1 wsl --install
 2.2 Setup user and login
 2.3 Update the linux system
  2.3.1 sudo apt-get update
  2.3.2 sudo apt-get upgrade
  2.3.3 sudo reboot
3.Install Anaconda(For managing environments)
 3.1  https://www.anaconda.com/download Linux Python 3.11 64-Bit (x86) Installer (1015.6 MB)
 3.2 Copy file to the linux system
 3.3 Install Anaconda 
  3.3.1 bash Anaconda-latest-Linux-x86_64.sh
  3.3.2 conda config --set auto_activate_base False 
 3.4 Create environments
  3.4.1 conda create -n myenv python=3.11
  3.4.2 conda activate myenv
4. Install CUDA
 4.1  https://developer.nvidia.com/cuda-too... (11.8)
 4.2 wget https://developer.download.nvidia.com...
 4.3 sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
 4.4 wget https://developer.download.nvidia.com...
 4.5 sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
 4.6 sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
 4.7 sudo apt-get update
 4.8 sudo apt-get -y install cuda
5. Install cuDDN
 5.1  https://developer.nvidia.com/rdp/cudn... (11.x, Local Installer for Ubuntu22.04 x86_64 (Deb) )
 5.2 Copy file to the linux system
 5.3 sudo dpkg -i cudnn-local-repo-$distro-8.x.x.x_1.0-1_amd64.deb
 5.4 sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
 5.5 sudo apt-get update
 5.6 sudo apt-get install libcudnn8=8.x.x.x-1+cudaX.Y
 5.7 sudo apt-get install libcudnn8-dev=8.x.x.x-1+cudaX.Y
 5.8 sudo apt-get install libcudnn8-samples=8.x.x.x-1+cudaX.Y
 5.9 sudo reboot 
6. pip install --upgrade pip
7. python3 -m pip install tensorflow[and-cuda]
8. pip install --ignore-installed --upgrade tensorflow==2.14
9. python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
10. conda install -c conda-forge jupyterlab
11. code .
12. VS Code WSL2 and Python plugin
```
</details>

## Translation Encoder-Decoder model Code Analysis

### **Encoder**

This section details the implementation of the provided encoder class, its architecture, hyperparameters, and memory usage.


**Network Architecture:**

The encoder is implemented as a subclass of `tf.keras.Model`, inheriting functionalities for building and training neural networks. It employs a two-layer architecture:

1. **Embedding Layer:**

   - The first layer is an `Embedding` layer with a vocabulary size `vocab_size` and an embedding dimension `embedding_dim`. This layer serves as a [lookup table](https://medium.com/analytics-vidhya/understanding-embedding-layer-in-keras-bbe3ff1327ce), mapping each word index in the source language sequence to a dense vector representation in `embedding_dim` space. This vector captures semantic information about the word and its relationship to other words in the vocabulary.
2. **Long Short-Term Memory (LSTM) Layer:**

   - The core component of the encoder is a single-stacked LSTM layer with `enc_units` memory units. LSTMs excel at handling sequential data by incorporating memory cells (LSTM cells) that can retain information for extended periods. This is crucial for capturing long-range dependencies within the source language sentence, where the meaning of a word can be influenced by words that appeared much earlier in the sequence.
   - The LSTM layer is configured with the following properties:
     - `return_sequences=True`: This ensures that the output of the LSTM layer retains the sequence dimension, providing an output for each element in the input sequence.
     - `return_state=True`: This instructs the LSTM layer to return not only the output sequence but also the hidden state and cell state at the final time step. These states encapsulate the accumulated knowledge about the entire processed sequence and are crucial for subsequent steps.
     - `recurrent_initializer='glorot_uniform'`: This defines the initialization method for the recurrent weights within the LSTM cell, promoting a healthy distribution of weight values and aiding gradient flow during training.

<details>
<summary>Functionalities</summary>

**Functionalities:**

The encoder class defines three core functionalities:

1. **Constructor (`__init__`):** This constructor initializes the encoder object, defining essential hyperparameters like `vocab_size`, `embedding_dim`, `enc_units`, and `batch_sz` (batch size). It also creates the `embedding` layer and the LSTM layer as its internal components.
2. **Forward Pass (`call`):**  This method serves as the forward pass through the encoder network. It takes two arguments:

   - `x`: The input tensor representing the source language sequence, typically a matrix of shape `(batch_size, sequence_length)`, where each element represents a word index.
   - `hidden`: A list containing two tensors, the initial hidden state and cell state for the LSTM layer. If not provided, the `initialize_hidden_state` method is called to create zeros for both states.
   - The method performs the following steps:
     - Embeds the input sequence using the `embedding` layer.
     - Passes the embedded sequence and the initial hidden state through the LSTM layer.
     - Returns three tensors:
       - Output sequence: This has the shape `(batch_size, sequence_length, enc_units)`, representing the output for each element in the processed sequence.
       - Hidden state (`h`): This has the shape `(batch_size, enc_units)`, capturing the context information at the final time step.
       - Cell state (`c`): This also has the shape `(batch_size, enc_units)`, containing additional information about the LSTM cell's memory.
3. `initialize_hidden_state`: This method creates a list containing two zero tensors, one for the initial hidden state and another for the initial cell state of the LSTM layer. The tensor shapes are both `(batch_size, enc_units)`.

**Hyperparameters (obtained from the code snippet):**

- `vocab_size`: Size of the vocabulary for the source language.
- `embedding_dim`: Dimensionality of the word embeddings.
- `enc_units`: Number of memory units in the LSTM layer.

</details>

<details>
<summary>Encoder Memory Analysis</summary>

**Memory Analysis (already in the code snippet (.ipynb)):**

The provided code snippet includes a detailed memory analysis of the encoder. This information is valuable for understanding the computational footprint of the model and optimizing resource allocation during training and deployment. The analysis breaks down the memory usage for the embedding layer, the LSTM layer, and the total model parameters.



The provided code snippet offers a detailed breakdown of the encoder's memory usage for each layer and the total model size.

**Embedding Layer:**

* Trainable weights: `encoder/embedding/embeddings:0 (34365, 256)`
* Memory usage: `33.5595703125 MB`

The embedding layer stores a large lookup table containing word vectors. The weight matrix has a shape of `(34365, 256)`, indicating:
    * `34365`: Vocabulary size (number of unique words)
    * `256`: Embedding dimension (size of each word vector)

**LSTM Layer:**

* Trainable weights:
    * `encoder/lstm/lstm_cell/kernel:0 (256, 4096)`
    * `encoder/lstm/lstm_cell/recurrent_kernel:0 (1024, 4096)`
    * `encoder/lstm/lstm_cell/bias:0 (4096,)`
* Memory usage:
    * Kernel weights: `encoder/lstm/lstm_cell/kernel:0 (256, 4096)` (dominant factor) - Memory usage not explicitly shown, but can be calculated as `(256 elements * 4096 elements per element * 4 bytes per element) / (1024^3 bytes per MB)`
    * Recurrent kernel weights: `encoder/lstm/lstm_cell/recurrent_kernel:0 (1024, 4096)` - Memory usage: `16.0 MB`
    * Bias: `encoder/lstm/lstm_cell/bias:0 (4096,)` - Memory usage: `0.015625 MB` (negligible)

The LSTM layer utilizes several weight matrices for processing sequences. The dominant memory usage comes from the kernel weights, which have a much larger size compared to recurrent kernel weights and bias.

**Total Model Memory:**

* Total model memory (trainable parameters only): `53.5751953125 MB`
* Total trainable parameters: `14,044,416`

The total memory usage reported here reflects the size of the trainable parameters in the model. This value might differ slightly from the sum of individual layer memory usages due to additional factors like optimizer state and model overhead.

**Key Takeaways:**

* The embedding layer consumes the most memory due to the large vocabulary size and embedding dimension.
* The LSTM layer's memory usage is primarily influenced by the kernel weights.
* Analyzing memory usage helps optimize hyperparameters and resource allocation for training and deployment.

**Note:** 
* This explanation assumes 4 bytes per element for weight matrices.
* The actual memory usage might vary depending on the system architecture and data types used.
</details>

### **Decoder**

This section presents the implementation of the Decoder class, outlining its architecture, hyperparameters, and memory usage.

**Network Architecture:**

The Decoder is designed as a subclass of `tf.keras.Model`, inheriting capabilities for constructing and training neural networks. It follows a multi-layer architecture:

1. **Embedding Layer:**

   - The first layer is an `Embedding` layer with a vocabulary size of `vocab_size` and an embedding dimension of `embedding_dim`. This layer performs word embedding, mapping each word index in the target language sequence to a dense vector representation in `embedding_dim` space. These vectors encapsulate semantic information crucial for decoding.

2. **Long Short-Term Memory (LSTM) Layer:**

   - The core component of the Decoder is a single LSTM cell with `dec_units` memory units. LSTMs are adept at capturing sequential dependencies, making them suitable for sequence generation tasks like language translation. The LSTM layer is wrapped within an attention mechanism to focus on relevant parts of the input sequence during decoding.

3. **Attention Mechanism:**

   - An attention mechanism enhances the LSTM's ability to attend to specific parts of the input sequence, facilitating more informed decoding. The attention mechanism is configured based on the specified `attention_type`, which can be either 'bahdanau' or 'luong'. Both mechanisms aim to align the decoder output with relevant parts of the input sequence, but they differ in their computational approaches.

4. **Dense Layer:**

   - The final layer is a `Dense` layer with a softmax activation function, producing probabilities for each word in the target vocabulary. This layer enables the model to generate the next word in the sequence based on the LSTM output and the attention context.

<details>
<summary>Functionalities</summary>

**Functionalities:**

The Decoder class encompasses the following functionalities:

1. **Constructor (`__init__`):** 
   - Initializes the decoder object with essential hyperparameters such as `vocab_size`, `embedding_dim`, `dec_units`, `batch_sz`, and `attention_type`.
   - Creates the embedding layer, LSTM cell, attention mechanism, and the decoder using TensorFlow Addons' `BasicDecoder`.

2. **Building RNN Cell (`build_rnn_cell`):** 
   - Constructs the fundamental recurrent cell for the decoder, incorporating the attention mechanism within an `AttentionWrapper`.

3. **Building Attention Mechanism (`build_attention_mechanism`):** 
   - Creates the attention mechanism based on the specified type ('bahdanau' or 'luong'), with parameters such as `dec_units`, `memory`, and `memory_sequence_length`.

4. **Building Initial State (`build_initial_state`):** 
   - Initializes the decoder's initial state, incorporating the encoder's final state to initiate the decoding process.

5. **Forward Pass (`call`):** 
   - Executes the forward pass through the decoder network, taking input tokens and the initial state as inputs. 
   - Embeds the input tokens, feeds them into the decoder, and retrieves the output sequence.

**Hyperparameters:**

- `vocab_size`: Size of the vocabulary for the target language.
- `embedding_dim`: Dimensionality of the word embeddings.
- `dec_units`: Number of memory units in the LSTM layer.
- `batch_sz`: Batch size for training data.
- `attention_type`: Type of attention mechanism used (Bahdanau or Luong).

</details>

<details>
<summary>Decoder Memory Analysis</summary>

**Memory Analysis:**

The provided code snippet includes a comprehensive memory analysis for the decoder, detailing the memory consumption of each layer and the total model size. The analysis covers trainable weights and their corresponding memory usage for the embedding layer, dense layer, LSTM cell, attention mechanism, and the overall model.

**Embedding Layer:**

- Trainable weights: `decoder/embedding_1/embeddings:0 (31733, 256)`
- Memory usage: `30.9892578125 MB`

**Dense Layer:**

- Trainable weights:
  - `decoder/basic_decoder/decoder/dense/kernel:0 (1024, 31733)`
  - `decoder/basic_decoder/decoder/dense/bias:0 (31733,)`
- Memory usage: `123.95703125 MB` (for kernel) + `0.12105178833007812 MB` (for bias)

**LSTM Cell:**

- Trainable weights:
  - `decoder/basic_decoder/decoder/attention_wrapper/lstm_cell_1/kernel:0 (1280, 4096)`
  - `decoder/basic_decoder/decoder/attention_wrapper/lstm_cell_1/recurrent_kernel:0 (1024, 4096)`
  - `decoder/basic_decoder/decoder/attention_wrapper/lstm_cell_1/bias:0 (4096,)`
- Memory usage: `20.0 MB` (for kernel) + `16.0 MB` (for recurrent kernel) + `0.015625 MB` (for bias) each

**Attention Mechanism (Luong):**

- Trainable weights: `LuongAttention/memory_layer/kernel:0 (1024, 1024)`
- Memory usage: `4.0 MB`

**Total Model Memory (Params only):**

- `203.08296585083008 MB`

**Note:** 
- The memory usage is calculated based on assumptions about the number of bytes per parameter element.
- Actual memory usage may vary depending on system architecture and data types used.

**Total model memory includes only trainable parameters and does not account for non-trainable weights.**

**Model Summary:**

- Total trainable parameters: `53,236,981`

**Model: "decoder"**
- Summary of individual layers and their respective output shapes.

**Keywords:** Decoder, Sequence-to-sequence Models, LSTM, Attention Mechanism, Embedding Layer, Memory Analysis.
</details>


## References (incomplete description)


## Keywords

| **Encoder** | **tf.keras.Model** | **Embedding Layer** | **LSTM Layer** | **Vocabulary Size** | **Embedding Dimension** | **LSTM** | **enc_units** | **Return Sequences** | **Return State** | **Recurrent Initializer** | **Functionalities** | **Constructor** | **Forward Pass** | **Initialize Hidden State** | **Hyperparameters** | **Memory Analysis** | **Trainable Weights** | **Total Model Memory** | **Total Trainable Parameters** | **Memory Usage** | **Attention Mechanism** | **Dense Layer** | **Bahdanau** | **Luong** | **Model Summary** |



Note: all the incomplete description will be completed in future, if you want to contribute, please do the PR