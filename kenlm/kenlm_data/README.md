# DATA FOR LANGUAGE MODEL FOR DECODING

Plese note that all the files contained in this folder are needed for using the tensorflow custop op `ctc_beam_search_decoder_with_lm` implemented by Mozilla [DeepSpeech project](https://github.com/mozilla/DeepSpeech)

In this you should place find all the files needed to use [KenLM](https://kheafield.com/code/kenlm/) language model:
- alphabet file : a file containing all the possible chars present in the language model (one per line)
- trie : all the vocabulary used for training the language model saved in a trie data structure 
- lm binary : the binary file of the language model

For quick out of the box solution you can find these files  in the folder `data/lm` of the [DeepSpeech project](https://github.com/mozilla/DeepSpeech).
All the files are generated using the train corpus of [LibriSpeech](http://www.openslr.org/12/)



