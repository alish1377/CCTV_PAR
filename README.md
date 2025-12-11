# CCTV_PAR

## Project Overview
Pedestrian Attribute Recognition (PAR) is recognized as one of the surveillance tasks for person re-identification in CCTVs. In this project, I'm working on a PAR algorithm called C2T_Net (Channel-Aware Cross-Fused Transformer-Style Networks), which is based on the ViT and Swin transformer architectures. The innovations of this project are summarized as:
1. **Dataset development**: We create a native dataset named [**Sharif_PAR**](https://github.com/SharifDeepLab/Sharif_PAR/tree/main) from surveillance cameras of Sharif University, a hospital, and a shopping market. For more information about the dataset, refer to the dataset link.
2. 
3. **Fine-tuning**: The basic C2T_Net model is trained on the [UPAR](https://openaccess.thecvf.com/content/WACV2024W/RWS/papers/Cormier_UPAR_Challenge_2024_Pedestrian_Attribute_Recognition_and_Attribute-Based_Person_Retrieval_WACVW_2024_paper.pdf) dataset. This large-scale dataset is adaptable to the culture and clothing types of people in foreign countries. So, we use 3 types of fine-tuning structures to tune the pretrained model to our dataset:
   - **Full fine-tuning**: Tune all layers, including Transformer, Channel-aware swinT, Cross-fusion, and head blocks
   - **Partial fine-tuning**: Tune the cross-fusion and head layers and freeze primary transformer-based blocks
   - **Head fine-tuning**: Tune the head block and freeze the transformer-based and fusion blocks
   - 
4. **Loss function optimization**: The native dataset is not balanced for all attributes. As the presence probability of many attributes is much lower than the absence of them, we use loss-based optimization approaches to consider positive samples(boxes include an attribute) more than negative samples(boxes don't include an attribute) in the model. The techniques are summarized as:
   - **Categorical Loss**: Introduces category-level supervision to better capture attribute group dependencies.
   - **Sample Weighting**: Adjusts loss contributions based on the imbalance of positive and negative label distributions.
   - **Logit Updating**: Refines predicted logits using recall information from positive and negative labels.
   - **GradNorm**: Learns separate weights for each attribute in addition to learning the base model weights.
   - **Focal Loss**: Reweights the loss to focus learning on harder, less frequent samples.

   
