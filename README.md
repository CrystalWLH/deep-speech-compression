# Listen! The Teacher is talking! 
## Create input data
Create data with:
    python -m utils.create_tfrecords.py

## Train teacher model
Train the teacher model:
    python model_main.py train -c ./configs/wav2letter_v1.config

## Create distilled knowledge
Save teacher logits to tfrecord file
    python -m utils.create_logits_tfrecord.py
    
## Train student network
Train student network with distilled knowledge from teacher
    python model_main.py train -c ./configs/wav2letter_v1.config
    


