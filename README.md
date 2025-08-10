# Vision-to-Text-Generation-via-Q-Former-and-a-GAN-Enhanced-Decoder

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/JustinSkD9/Vision-to-Text-Generation-via-Q-Former-and-a-GAN-Enhanced-Decoder.git
    cd Vision-to-Text-Generation-via-Q-Former-and-a-GAN-Enhanced-Decoder
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run prepare_dataset.py to download and prepare the Dataset subset (using MS-COCO 2017 training set)
    ```bash
    python prepare_dataset.py

    options:
    -h, --help            show this help message and exit    
    --regenerate          Force regenerate dataset even if   
                            it exists
    --subset-size SUBSET_SIZE
                            Number of images in the subset     
                            (default: 10000)
    ```
4. Run main.py to train, and display some examples at the end.
    ```bash
    python main.py

    options:
    -h, --help            show this help message and exit
    --use-gan             Enable GAN training after supervised training
    --skip-supervised     Skip supervised training (use existing checkpoint)
    --re-extract          Re-extract CLIP features
    --checkpoint          Load existing supervised checkpoint if available
    --gan-checkpoint      Load existing GAN checkpoint if available
    --supervised-epochs SUPERVISED_EPOCHS
                            Number of supervised training epochs
    --gan-epochs GAN_EPOCHS
                            Number of GAN training epochs
    --warmup-epochs WARMUP_EPOCHS
                            Number of warmup epochs (T5 frozen)
    --examples EXAMPLES   Number of examples to generate
    ```
5. Run prepare_datase.py to display the loss and metrics plots over epoches.
    ```bash
    python prepare_dataset.py
    ```