This folder includes all the codes, intermediate files and final results of LDA.
Running the code can generate LDA topics and plot the words distribution.

overall_topics includes the 7 topics with top10 words.
title_topics includes the 7 topics with top10 words for all the titles.
problem_topics includes the 7 topics with top10 words for all the problems.
implication_topics includes the 7 topics with top10 words for all the implications.
workaround_topics	includes the 7 topics with top10 words for all the workarounds.

Topic distribution is the file containing the topic distribution probability values for title, problem, implication and workaround separately. topicchart.png is the bar chart for those values.

topics_lda.py is the main src code for LDA. It will extract the topics from newdata.tsv, distribution values and also output plot_topics.png under the same directory. 
