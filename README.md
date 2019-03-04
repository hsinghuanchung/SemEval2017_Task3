# SemEval2017_Task3 
Term project of NTU CSIE Information Retrieval and Extraction, 2018 Fall

## Project Structure 
This repository provides four method for ["SemEval-2017 Task 3: Community Question Answering"](http://aclweb.org/anthology/S17-2003).
* Bilateral Matching
* BM25
* Siamese CNN
* Siamese CNN + RNN

You may enter each folder to examine how to run the code.
Detailed description of the methods and their results are in Report.pdf.

## Task Description
### Subtask A: Question-Comment Similarity
Given a question and the first 10 comments in its question thread, rerank these 10 comments according to their relevance with respect to the question.

### Subtask B: Question-Question Similarity
Given a new question (aka original question) and the set of the first 10 related questions (retrieved by a search engine), rerank the related questions according to their similarity with the original question.

### Subtask C: Question-External Comment Similarity
Given a new question (aka the original question) and the set of the first 10 related questions (retrieved by a search engine), each associated with its first 10 comments appearing in its thread, rerank the 100 comments (10 questions x 10 comments) according to their relevance with respect to the original question.
