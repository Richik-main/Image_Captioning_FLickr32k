# Image_Captioning_FLickr32k
This is a image captioning project from sratch 

Here for the alignment model i will be using this architecture

```plaintext
         +-------------------+        +-------------------+
         |    Input Image    |        |    Input Caption  |
         +-------------------+        +-------------------+
                    |                           |
           [Preprocessing]             [Tokenization]
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
                        |  Similarity Matrix|
                        |(cosine similarity)|
                        +------------------+
                                |
                        +------------------+
                        | Contrastive Loss |
                        +------------------+
```
