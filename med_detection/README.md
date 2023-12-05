## Medical Detection

As part of our data preprocessing, we noticed that some questions in the Yahoo L6
dataset, even in the health category, were not related to medical questions. We
therefore decided to use NLI as a way to filter out questions that were not
related to medical questions.

To classify questions, run the following command:

```
python med_detection/nli_medical_questions.py
```

after modifying the input and output paths in the script.