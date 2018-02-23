@title[Main slide]

## Natural Language Processing
<span style="font-size:0.6em; color:gray">Máster Big Data Science (UVa)</span> |
<span style="font-size:0.6em; color:gray">Fernando Rabanal Presa</span>

---
#### Disclaimer

<br><hr>
All materials provided here reflect my own views and not those of my employer.
<hr>
Please, do not take my opinions too seriously as I tend to be wrong more times than expected (on average) every single day.

---
#### Who am I?

![CatGIF](https://i.giphy.com/media/JIX9t2j0ZTN9S/giphy-downsized.gif)

<span style="color:gray; font-size:0.6em">Linkedin: [fernandorabanal](https://www.linkedin.com/in/fernandorabanal/)</span> |
<span style="color:gray; font-size:0.6em">Email: [frabanalpresa@gmail.com](mailto:frabanalpresa@gmail.com)</span>
<br>

@fa[arrow-down]

+++

- Telecommunications Engineer -- UVa
- MSc. Multimedia and Communications (Signal Theory) -- UC3M
- *Kaggle* Master level (Competition winner)
<br>

@fa[arrow-down]

+++

- Data Scientist in *NeoMetrics*/*Accenture Advanced Analytics*
- Data Scientist in *Touchvie*/*Dive*
- Data Scientist in *Vodafone Group*

@fa[arrow-down]

+++

#### But it is more interesting to see what can you do

- Complete a **MSc** program in Data Science (you're on the right track!)
- Complement your knowledge with some MOOC programs |
- Practice with different problems |
- Enter Data Science competitions |
- **Provide value to your company** |

---

### What is Natural Language Processing?

> Are there imaginable digital computers which would do well in the *imitation game*?

<br>
<div style="text-align: right"><span style="color:gray; font-size:0.5em">Alan Turing, "Computing Machinery and Intelligence", 1950</span></div>

@fa[arrow-down]

+++

> Natural language processing (NLP) can be defined as the ability of a machine to analyze, understand, and generate human speech. The goal of NLP is to make interactions between computers and humans feel exactly like interactions between humans and humans.

<br>
<div style="text-align: right"><span style="color:gray; font-size:0.5em">[NeoSpeech, 2013](https://blog.neospeech.com/what-is-natural-language-processing/)</span></div>

@fa[arrow-down]

+++

### Applications:

**Chatbots**

![Image-Absolute](https://cdn-images-1.medium.com/max/1600/1*hdUgYLkAbzzMCRzOLsrnEA.gif)

<span style="color:gray; font-size:0.5em">Ordering with Tacobot in <b>Slack</b>. Original blog post by  [chatbotsmagazine](https://chatbotsmagazine.com/11-examples-of-conversational-commerce-57bb8783d332)</span>

@fa[arrow-down]

+++

**Spam filtering**

![Image-Absolute](https://images.pexels.com/photos/265651/pexels-photo-265651.jpeg?w=600&h=480&auto=compress&cs=tinysrgb)

<span style="font-size:0.5em; color:gray">Analyzes header</span> |
<span style="font-size:0.5em; color:gray">Explores content</span> |
<span style="font-size:0.5em; color:gray">Checks for spamming rules</span>

@fa[arrow-down]

+++

**Text classification**

![Image-Absolute](https://cdn-images-1.medium.com/max/1600/1*ljCBykAJUnvaZcuPYwm4_A.png)

<span style="color:gray; font-size:0.5em">[Towards Data Science](https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a) article on Text classification with scikit-learn and NLTK</span>


@fa[arrow-down]

+++

**Text tagging (clustering)**

![Image-Relative](assets/images/flickr_tags_screenshot.png)

<span style="color:gray; font-size:0.5em"><b>Flickr</b> [tags](https://www.flickr.com/photos/tags) page.</span>

@fa[arrow-down]

+++

**News summarization**

---?image=assets/images/agolo_screenshot.png&size=75% 75%

<span style="color:gray; font-size:0.5em">[Agolo](https://www.agolo.com) is a commercial news summarizer software.</span>


@fa[arrow-down]

+++

## Algorithmic trading

<canvas data-chart="line">
{
 "data": {
  "labels": ["Sep 22","Sep 23","Sep 24","Sep 25","Sep 26","Sep 27","Sep 28", "Sep 29", "Sep 30"],
  "datasets": [
   {
    "data":[0.057156, 0.058499, 0.056775, 0.06, 0.060428, 0.074999, 0.104, 0.074786, 0.069976],
    "backgroundColor":"rgba(20,220,220,.8)"
   }
  ]
 },
 "options": { "responsive": "true" }
}
</canvas>

<span style="color:gray; font-size:0.5em">Cryptocurrencies ZEC/BTE exchange rate</span>

---

NLP usual workflow
<!-- ?code=src/workflow.py&lang=python&title=NLP usual workflow

@[1,3-6](Text preprocessing)
@[8-18](Text modeling)
@[19-28](Extract insights) -->

---

## Text preprocessing

- Text is usually composed of words and characters in the real world.
- Redundancy and use of low informative words (linkers, prepositions...) is used in text to provide context.
- Variations of the same word (dog, dogs, puppies...) are used also to provide small differences in meaning, but
they usually do not change the overall meaning of a document.
- Machine Learning algorithms do usually take numeric tensors as inputs.

<hr>

- A document needs to be preprocessed to conform a viable numeric representation that represents it respect to the
problem to be solved.

---

## Text cleaning

?code=src/text_cleaning.py&lang=python&title=NLP usual workflow

@[1,3-6](Strip markup tags)
@[8-18](Remove URLs)
@[19-28](Remove punctuation signs)

@fa[arrow-down]

+++

## Text cleaning (II)

?code=src/text_cleaning.py&lang=python&title=NLP usual workflow

@[1,3-6](Normalize to lowercase characters)
@[8-18](Remove non-informative words, expressions...)
@[19-28](Remove non-alphanumeric characters)


---

## Lemmatization

> Lemmatization is the process of grouping together the inflected forms of a word so they can be analysed
as a single item, identified by the word's **lemma**, or dictionary form.
> Wikipedia (last access, Feb. 11, 2018)

- It is a linguistic process, so language knowledge is used and assumed in this process.

@fa[arrow-down]

+++

## Lemmatization examples

```python
lemmatize('better') = 'good'
lemmatize('walking') = 'walk'
lemmatize('meeting') = ['meet', 'meeting']
```

@[1,2](Lemmatization assumes language knowledge.)
@[3](And more than one lemma can result from process.)

---

## Stemming

> Stemming is the process of reducing inflected (or sometimes derived) words to their word stem,
base or root form—generally a written word form.
> Wikipedia (last access, Feb. 11, 2018)

- It does not assume any linguistics involved, it is enough to map related words to a common root, even if this root
is not a valid root morphologically speaking.

@fa[arrow-down]

+++

## Stemming examples

```python
lemmatize('better') = 'better'
lemmatize('walking') = 'walk'
lemmatize('meeting') = 'meet'
```

@[1,2](No linguistics assumed.)
@[3](Just morphological root is returned.)

---

## Stopwords removal

![Image-Absolute](http://www.michaeljgrogan.com/wp-content/uploads/2017/10/wordcloud-450x325.png)

- A list of non-informative words (*stopwords*) can be tailored for each language of interest.
- Topic-specific stopwords can be added to the generic list to separate the 'signal' from the 'noise' in a specific
problem.
- Stopwords list is usually removed after lowercase normalization to avoid exploring different cases related to
unnormalized documents.

<span style="color:gray; font-size:0.5em">tidytext: Word Clouds and Sentiment Analysis in R, [Michael Grogan](http://www.michaeljgrogan.com/tidytext-word-clouds-sentiment-r/)</span>

@fa[arrow-down]

+++

## Stopwords example (Spanish)

<table>
  <tr>
    <td>un</td>
    <td>una</td>
    <td>unas</td>
    <td>unos</td>
  </tr>
  <tr class="fragment">
    <td>aquél</td>
    <td>aquéllos</td>
    <td>aquélla</td>
    <td>aquéllas</td>
  </tr>
  <tr class="fragment">
    <td>ante</td>
    <td>bajo</td>
    <td>de</td>
    <td>desde</td>
  </tr>
  <tr class="fragment">
    <td>y</td>
    <td>o</td>
    <td>mas</td>
    <td>sino</td>
  </tr>
  <tr class="fragment">
    <td>muy</td>
    <td>poco</td>
    <td>con</td>
    <td>sin</td>
  </tr>
</table>

---

## Tokenization

![Image-Absolute](assets/tokenization_example.png)

> Tokenization is the process of demarcating and possibly classifying sections of a string of input characters.
> Wikipedia (last access, Feb. 11, 2018)

- Tokenization is usually the result of applying heuristics to a document to split a single string into a set
of tokens, which usually pass onto a different form of preprocessing.

<span style="color:gray; font-size:0.5em">Tokenization example</span>

@fa[arrow-down]

+++

## Tokenization examples

- Tokens obtained by splitting sentences by spaces.
- Separate numbers from letters to be treated as different tokens.
- Words with hyphens (-) can be splitted into two tokens (punctuation removal has to account for this case beforehand).
<br>

How to choose best tokenization process:

<table>
  <tr>
    <td>Manual process</td>
    <td>Data driven</td>
  </tr>
  <tr class="fragment">
    <td>Task-specific</td>
    <td>Language-specific</td>
  </tr>
</table>


---

## From text to numbers

![Image-Absolute](assets/text_to_numbers.png)


---

## Bag of Words (BoW)

Intends to obtain a Vector Space representation of a document, originally in text domain.

- Madrid looks sunnier than usual this spring. |
- Real Madrid plays against Atlético Madrid tonight.

<br>

<table>
  <tr>
    <td>Manual process</td>
    <td>Data driven</td>
  </tr>
  <tr class="fragment">
    <td>Task-specific</td>
    <td>Language-specific</td>
  </tr>
</table>


@fa[arrow-down]

+++

## N-gram representation

> *n-gram* is a contiguous sequence of n items from a given sample of text or speech. The items can be phonemes,
syllables, letters, words or base pairs according to the application.
> Wikipedia (last access, Feb. 11, 2018)

<hr>
Real Madrid plays against Atlético Madrid tonight.

<table>
  <tr>
    <th>Unigram (size 1)</th>
    <th>Bigram (size 2)</th>
  </tr>
  <tr class="fragment">
    <td>Madrid</td>
    <td>Real Madrid</td>
  </tr>
  <tr class="fragment">
    <td>Madrid</td>
    <td>Atlético Madrid</td>
  </tr>
  <tr class="fragment">
    <td>plays</td>
    <td>plays against</td>
  </tr>
</table>

@fa[arrow-down]

+++

## TF-IDF

Term Frequency: different measures of raw frequency.

`\begin{eqnarray}
tf(t,d) & = & f(t,d) \nonumber\\
tf(t,d) & = & 1+log(f(t,d)) ~,~ (log(0) \coloneqq 0) \nonumber\\
tf(t,d) & = & \dfrac{f(t,d)}{\max \lbrace f(w,d) : w \in D \rbrace} \nonumber
\end{eqnarray}`

<br>

Inverse Document Frequency: commonality of the term in the whole collection of documents.
`\begin{equation*}
idf(t,D) = \log \dfrac{|D|}{|\lbrace d \in D : t \in d \rbrace|}
\end{equation*}`

---

## Word embeddings

![Image-Absolute](assets/umap_embedding.png)

- [GloVe](https://nlp.stanford.edu/projects/glove/)
- [Word2Vec / Doc2Vec](https://arxiv.org/pdf/1301.3781.pdf?)

<span style="color:gray; font-size:0.5em">Visualization of 3M words from GoogleNews dataset as embedded by  [UMAP](https://arxiv.org/pdf/1802.03426.pdf)</span>

---

## Sentiment analysis

![Image-Absolute](https://???.jpg)

<span style="color:gray; font-size:0.5em">the <b>Daftpunktocat-Guy</b> by [jeejkang](https://github.com/jeejkang)</span>

---

## Clustering

![Image-Absolute](https://???.jpg)

<span style="color:gray; font-size:0.5em">the <b>Daftpunktocat-Guy</b> by [jeejkang](https://github.com/jeejkang)</span>

@fa[arrow-down]

+++

## Clustering algorithms

- K-means
- LSI
- LDA

---

## Supervised problems

![Image-Absolute](https://???.jpg)

<span style="color:gray; font-size:0.5em">the <b>Daftpunktocat-Guy</b> by [jeejkang](https://github.com/jeejkang)</span>

@fa[arrow-down]

+++

## Supervised algorithms

- Linear/logistic regression
- Neural networks
- K-Nearest Neighbors

---

## Generative modeling

![Image-Absolute](https://ddg-mjesip8vchewh1dsl.stackpathdns.com/assets/landing/img/gallery/4.jpg)

<span style="color:gray; font-size:0.5em"><b>Deep Dream</b> example by [deepdreamgenerator](https://deepdreamgenerator.com)</span>


@fa[arrow-down]

+++

## Generative models

![Image-Absolute](https://deeplearning4j.org/img/srn_elman.png)

- [LSTM](https://deeplearning4j.org/lstm.html)
- [GANs](https://deeplearning4j.org/generative-adversarial-network)

<span style="color:gray; font-size:0.5em"><b>Recurrent Net</b> by [Elman](https://deeplearning4j.org/lstm.html)</span>

---

## Software modules


---

## R packages: tm

![Image-Absolute](assets/tm_documentation.png)

<table>
  <tr>
    <th>Pros</th>
    <th>Cons</th>
  </tr>
  <tr class="fragment">
    <td>Classic solution</td>
    <td>Low-level package</td>
  </tr>
  <tr class="fragment">
    <td>Stable</td>
    <td></td>
  </tr>
  <tr class="fragment">
    <td>Extensible with plugins</td>
    <td></td>
  </tr>
</table>

<span style="color:gray; font-size:0.5em">[tm](https://cran.r-project.org/web/packages/tm/tm.pdf) documentation first page.</span>

@fa[arrow-down]

+++

## R packages: tidytext

![Image-Absolute](https://www.tidytextmining.com/images/cover.png)

<table>
  <tr>
    <th>Pros</th>
    <th>Cons</th>
  </tr>
  <tr class="fragment">
    <td>Great documentation and textbook</td>
    <td>Small support community</td>
  </tr>
  <tr class="fragment">
    <td>High-level algorithms</td>
    <td></td>
  </tr>
  <tr class="fragment">
    <td>Pipelines supported</td>
    <td></td>
  </tr>
</table>

<span style="color:gray; font-size:0.5em">[tidytext](https://www.tidytextmining.com/) book cover.</span>

---

## Python modules: NLTK

![Image-Absolute](http://www.nltk.org/_images/tree.gif)

<table>
  <tr>
    <th>Pros</th>
    <th>Cons</th>
  </tr>
  <tr class="fragment">
    <td>Myriad of resources</td>
    <td>Lacks newest algorithms</td>
  </tr>
  <tr class="fragment">
    <td>Complete classic NLP</td>
    <td></td>
  </tr>
  <tr class="fragment">
    <td>Standard solution for NLP</td>
    <td></td>
  </tr>
  <tr class="fragment">
    <td>Huge community</td>
    <td></td>
  </tr>
</table>

<span style="color:gray; font-size:0.5em">Sentence parsed with [NLTK](http://www.nltk.org/)</span>

@fa[arrow-down]

+++

## Python modules: gensim

![Image-Absolute](https://radimrehurek.com/gensim/_static/images/logo-gensim.png)

<table>
  <tr>
    <th>Pros</th>
    <th>Cons</th>
  </tr>
  <tr class="fragment">
    <td>High-level algorithms</td>
    <td>Learning curve</td>
  </tr>
  <tr class="fragment">
    <td>Some cutting-edge advances</td>
    <td>Difficult to grasp details</td>
  </tr>
  <tr class="fragment">
    <td>In-disk capabilities</td>
    <td></td>
  </tr>
  <tr class="fragment">
    <td>Great documentation and examples</td>
    <td></td>
  </tr>
  <tr class="fragment">
    <td>Distributed computing</td>
    <td></td>
  </tr>
</table>

<span style="color:gray; font-size:0.5em">[Gensim](https://radimrehurek.com/gensim/) logo.</span>

---

## Commercial software

Just a small selection of them are:

<table>
  <tr class="fragment">
    <td>IBM Watson</td>
    <td>Google Cloud Translation</td>
    <td>NVivo</td>
    <td>IBM SPSS Text Analytics</td>
  </tr>
  <tr class="fragment">
    <td>SAS High Performance Text Mining</td>
    <td>Microsoft Language Understanding Intelligent Service</td>
    <td>OdinText</td>
    <td>Amazon Comprehend</td>
  </tr>
</table>


---

## Next steps

<span style="font-size:0.6em; color:gray">Where are we now?</span> |
<span style="font-size:0.6em; color:gray">What else is yet to come?</span>

<table>
  <tr>
    <th>Problems</th>
    <th>Current solution</th>
    <th>Near future</th>
  </tr>
  <tr class="fragment">
    <td>Spam classification</td>
    <td>Supervised algorithms (~99%)</td>
    <td>--</td>
  </tr>
  <tr class="fragment">
    <td>Chatbots</td>
    <td>Generative models</td>
    <td>Improved generative models</td>
  </tr>
  <tr class="fragment">
    <td>Text summarizer</td>
    <td>Document embeddings</td>
    <td>Local/global embeddings</td>
  </tr>
  <tr class="fragment">
    <td>Documents semantics</td>
    <td>Document embeddings</td>
    <td>Semantic modelling</td>
  </tr>
</table>


@fa[arrow-down]

+++

## Online learning

![Image-Absolute](https://d1z75bzl1vljy2.cloudfront.net/kitchen-sink/octocat-privateinvestocat.jpg)

<span style="color:gray; font-size:0.5em">the <b>Daftpunktocat-Guy</b> by [jeejkang](https://github.com/jeejkang)</span>


@fa[arrow-down]

+++

## Active learning

![Image-Absolute](assets/active_learning_example.png)

<span style="color:gray; font-size:0.5em">Bryan Pardo, EECS 395/495 Modern Methods in Machine Learning, Spring 2010</span>

@fa[arrow-down]

---

## Online resources

- [RDataMining](http://www.rdatamining.com/docs)
- [Datacamp](https://www.datacamp.com/community/blog/text-mining-in-r-and-python-tips) tips for Text Mining projects in R/Python.
- [gensim](http://radimrehurek.com/gensim/wiki.html) documentation page (with tutorials).

@fa[arrow-down]

+++

## More online resources

- [tidytext](http://tidytextmining.com/) R package textbook.
- [NLTK](http://www.nltk.org/book/) free textbook for NLP with Python.
<br>
- [Information Retrieval](https://nlp.stanford.edu/IR-book/) textbook (Free)
- [Foundations of Statistical Natural Language Processing](https://nlp.stanford.edu/fsnlp/promo/) textbook (Not free)
