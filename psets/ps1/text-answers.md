# Deliverable 1.3

Why do you think the type-token ratio is lower for the dev data as compared to the training data?

(Yes the dev set is smaller; why does this impact the token-type ratio?)

Because the dev set is smaller, you are likely to see more unique types (ie those that only appeared once or a few times
in the corpus). Because word count distributions follow a power law distribution, additional samples tend to cluster
on words that have already been seen, increasing the number of tokens relative to types. As an extreme example:
"She sells seashells by the sea shore" has a token-type ratio of 1, ie each word is unique. This is easy with a small
corpus, but becomes impossible with a larger one, given the syntactical necessity of many stopwords.


# Deliverable 3.5

Explain what you see in the scatter plot of weights across different smoothing values.

The scatter plot shows the weights of similar features using very different smoothing values. For the most part, weights
for both values are similar, ie the scatter plot is linear, EXCEPT for features of large negative weight, ie those that
appear very infrequently, or are even unique. Smoothing essentially provides a larger "floor" to the weights of those
and while a smoothing parameter of 0.001 creates (on this dataset) a minimum weight of ~-19, the larger smoothing
parameter of 10, allows creates a minimum weight of ~-10.

# Deliverable 6.2

Now compare the top 5 features for logistic regression under the largest regularizer and the smallest regularizer.
Paste the output into ```text_answers.md```, and explain the difference. (.4/.2 points)


# Deliverable 7.2

Explain the new preprocessing that you designed: why you thought it would help, and whether it did.

# Deliverable 8

Describe the research paper that you have chosen.

- What are the labels, and how were they obtained?
- Why is it interesting/useful to predict these labels?  
- What classifier(s) do they use, and the reasons behind their choice? Do they use linear classifiers like the ones in this problem set?
- What features do they use? Explain any features outside the bag-of-words model, and why they used them.
- What is the conclusion of the paper? Do they compare between classifiers, between feature sets, or on some other dimension? 
- Give a one-sentence summary of the message that they are trying to leave for the reader.
