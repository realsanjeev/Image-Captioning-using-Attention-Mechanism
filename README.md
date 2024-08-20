# Image Captioning using Attention Mechanism

Image captioning is the task of generating descriptive and relevant textual captions for images automatically. The attention mechanism allows the model to focus on specific parts of the image while generating captions, resulting in more accurate and contextually meaningful captions.

## Architecture Overview

This project implements an **Encoder-Decoder architecture with Attention Mechanism** for generating image captions. The model combines Convolutional Neural Networks (CNN) for visual feature extraction and Recurrent Neural Networks (RNN) for sequential caption generation.

### Architecture Components

#### 1. **CNN Encoder (Feature Extractor)**
- **Model**: InceptionResNetV2 (pre-trained on ImageNet)
- **Configuration**: 
  - `include_top=False` - Removes the final classification layer
  - `weights="imagenet"` - Uses pre-trained weights
  - `trainable=False` - Frozen during training (transfer learning)
- **Input**: Images resized to `299×299×3`
- **Output**: Feature maps of shape `(8, 8, 1536)`
- **Processing**:
  - Feature maps are reshaped to `(64, 1536)` - 64 spatial locations with 1536 features each
  - Dense layer projects features to attention dimension (512) with ReLU activation
  - Final encoder output: `(64, 512)` representing 64 image regions

#### 2. **RNN Decoder with Attention**
- **Embedding Layer**: Converts word tokens to dense vectors of dimension 512
- **GRU (Gated Recurrent Unit)**:
  - Hidden dimension: 512
  - `return_sequences=True` - Returns full sequence for attention
  - `return_state=True` - Returns hidden state for inference
- **Attention Mechanism**:
  - Type: Bahdanau-style attention (additive attention)
  - Computes context vector by attending to encoder outputs
  - Allows the model to focus on relevant image regions while generating each word
- **Architecture Flow**:
  1. Word embeddings pass through GRU
  2. Attention layer computes context from GRU output and encoder features
  3. GRU output and context vector are added
  4. Layer normalization is applied
  5. Dense layer projects to vocabulary size for word prediction

#### 3. **Text Processing Pipeline**
- **Tokenization**: `TextVectorization` layer with:
  - Vocabulary size: 1000 tokens
  - Max sequence length: 65 tokens
  - Standardization: Lowercase conversion and punctuation removal
  - Special tokens: `<start>` and `<end>` markers
- **Word-to-Index Mapping**: `StringLookup` layers for bidirectional conversion

### Training Configuration

- **Dataset**: COCO Captions (640 images for demonstration)
- **Batch Size**: 32
- **Loss Function**: Sparse Categorical Crossentropy with masking for padding tokens
- **Optimizer**: Adam
- **Epochs**: 5 (in the example)
- **Image Preprocessing**: Normalization to [0, 1] range

### Inference Strategy

The model uses **probabilistic sampling** during inference:
1. Encoder processes the input image to extract features
2. Decoder starts with `<start>` token and zero GRU state
3. At each step:
   - Top-10 probable words are selected
   - One word is randomly sampled from this distribution
   - Sampled word becomes input for next step
4. Generation continues until `<end>` token or max length is reached

This sampling strategy produces diverse captions for the same image across multiple runs.

### Key Features

✅ **Transfer Learning**: Leverages pre-trained InceptionResNetV2 for robust visual features  
✅ **Attention Mechanism**: Enables interpretable, context-aware caption generation  
✅ **GRU Architecture**: Efficient recurrent processing with gating mechanisms  
✅ **Masked Loss**: Proper handling of variable-length sequences  
✅ **Probabilistic Decoding**: Generates diverse and creative captions
