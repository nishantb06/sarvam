# Forced CTC allignment for Indic languages (Hindi)
##### This is code for [hiring challenge](https://colab.research.google.com/drive/1EiiLTf5zB8Jm2PxdU3H20rWUr40FrsGM?usp=sharing) for Sarvam AI Speech team, [linkedin post](https://www.linkedin.com/posts/sarvam-ai_sarvam-ai-activity-7195640684052197378-PfEt?utm_source=share&utm_medium=member_desktop)

Allignment is problem frequently seen in Text-to-speech or speech-to-text tasks where we need to accurately map text to its phonetic representation in the audio. 

Simply speaking we need the starting and ending timestamp of every word in the transcript of when it is spoken in the audio. (These timestamps can be on a character basis as well and on sentence level as well, depending on the use case)
Note that for forced allignement, we need to already have the transcript of the audio file. 

This problem becomes difficult for Indic languages for a couple of reasons
- as we dont have strong foundational audio models for all Indic languages. Here I have used Wave2vec2-Hindi trained by ai4bharat to perform allignment on the data I scraped from a public website. 
- effective tokenization is difficult for Indic languages.

### Dataset 
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/nishantbhansali/new-testament-readings-in-hindi-260-chapters)




