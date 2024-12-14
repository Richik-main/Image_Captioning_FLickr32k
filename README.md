# Image Captioning FLickr32k

This is an image and embedding alignment project which can be used to retrieve images using texts.
We are fine-tuning the **BERT and ViT transformer** 

`main_ca_old.py` has the one which will run to get Cross attention and with cls embedding 
to get the mean embedding you can change the `train_ca_mean.py` to `trainerca.py`

`main.py` has the one which does not have cross attention layer the diagram below depicts its architecture.
Similarly, here we can change the `train_ca_mean.py` to `trainerca.py`

`main_ca_LoRA` is the file which can be run to finetune using LoRA. Peft technique

`main_ca` this file has the cross attention layer with the cls embedding.


Here for the alignment model I will be using this architecture

```plaintext
         +-------------------+        +-------------------+
         |    Input Image    |        |    Input Caption  |
         +-------------------+        +-------------------+
                    |                           |
             [Preprocessing]              [Tokenization]
                    |                           |
                    V                           V
         +-------------------+        +-------------------+
         |  Image Encoder    |        |  Text Encoder     |
         |  (e.g., ViT)      |        |  (e.g., BERT)     |
         +-------------------+        +-------------------+
                    |                           |
         [CLS Token or Global]         [CLS Token Output]
           Embedding Output                  Embedding
                    |                           |
                    V                           V
         +-------------------+        +-------------------+
         | Projection Head   |        | Projection Head   |
         |  (Linear Layer)   |        |  (Linear Layer)   |
         +-------------------+        +-------------------+
                    |                           |
           [Normalize Embeddings]      [Normalize Embeddings]
                    |                           |
                    +-----------+   +-----------+
                                |   |
                                V   V
                        +------------------+
                        | Similarity Matrix |
                        |(cosine similarity)|
                        +------------------+
                                |
                        +------------------+
                        | Contrastive Loss |
                        +------------------+
```
