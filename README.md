# DSCI_410_Anonymization_Script

# Python Script for Data Processing and Analysis

This README provides details about a Python script that utilizes various libraries to perform data processing, natural language processing (NLP), and visualization tasks. The script is designed to be a comprehensive tool for analyzing text data and generating synthetic data for testing purposes.

## Requirements

- Python 3.11
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `re` (part of the Python Standard Library)
  - `os` (part of the Python Standard Library)
  - `spacy`
  - `scikit-learn`
  - `Faker`

## Installation

Before running the script, ensure that you have Python installed on your system. You can then install the required libraries using pip:

```bash
pip install numpy pandas matplotlib spacy scikit-learn Faker
```

For `spacy`, you will also need to download a language model. For English, you can download the default model with:

```bash
python -m spacy download en_core_web_sm
```

## Script Overview

The script includes the following key functionalities:

1. **Data Manipulation**: Uses `pandas` and `numpy` for data handling and computations.
2. **Text Processing**: Utilizes `spacy` for advanced natural language processing tasks.
3. **Feature Extraction**: Implements `TfidfVectorizer` from `scikit-learn` to convert text data into a matrix of TF-IDF features.
3. **Similarity Measurement**: Employs cosine similarity from `scikit-learn` to compare text documents.
4. **Data Visualization**: Uses `matplotlib` to create visual representations of the data.
5. **Synthetic Data Generation**: Uses `Faker` to generate synthetic data for testing and simulation purposes.

## Usage

To use the script, simply import the modules at the top of your Python file as shown in the import section of this README. Below is an example of how to use these imports effectively:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faker import Faker

fake = Faker()
nlp = spacy.load('en_core_web_sm')  # Loading the English model

# Example usage with Faker
print(fake.name())

# Example NLP Processing
doc = nlp("This is an example sentence for NLP processing.")
for token in doc:
    print(token.text, token.lemma_, token.pos_)

# Example data visualization
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()
```

## Contributing

This script is part of a class project and is intended for educational purposes. If you are a fellow student or educator and wish to contribute, improve, or adapt the script for academic purposes, please feel free to do so. Contributions should align with academic integrity and the objectives of the course.

## License

This script is released under the GNU General Public License, Version 3. This license allows free use, modification, and distribution, but it must accompany any significant pieces of the redistributed code to ensure future users can benefit from modifications.

---
