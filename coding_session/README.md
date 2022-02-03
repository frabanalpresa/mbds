# Coding session
## Purpose

In this session, we solved a NLP Use Case related to news categorization. We used for that purpose the [BBC News Summary](https://www.kaggle.com/pariza/bbc-news-summary) dataset. The session took place on January 25, 2022.

You'll find all materials (except the dataset, which can be downloaded from the provided link) here to replicate and analyze the code:
- Empty notebook (`live_session.ipynb`)
- Classroom solutions, quick and dirty, without any additional comments (`live_session - classroom.ipynb`)
- Complete solutions, with comments (`live_session - complete.ipynb`)

## Preparation instructions

Before you dive into the code, you'll need to set up the virtual environment using Anaconda. In your terminal (or Anaconda Prompt, if you are in Windows), you can type:

```bash
conda env create -f environment.yml
conda activate nlp-mbds
python -m spacy download en_core_web_lg
```

And, if the environment is created and the spaCy model downloaded, we are ready to go! Just type:

```bash
jupyter lab
```

and start coding!