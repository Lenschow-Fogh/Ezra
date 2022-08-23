# Mitigation of stigma, prejudice anddiscrimination during early stages ofrecruitment using machine learning

## Abstract
Due to demographic characteristics and socioeconomic status, applicants are facing discrimination in the
early stages of job recruitment due to conscious or subconscious stigmatization by the employer. Not
only may applicants discarded by the employer based on demographic characteristics, but leading up to
pursuing a job, how a job advertisement is phrased with respect to gendered wording can affect pursuit
intentions. Studies find that race and ethnicity of the applicant can entail that the applicant has to
apply for double as many jobs as an applicant appertaining a majority, to receive the same amount of
callbacks. Studies also find, that if a job application or advertisement contains strong gendered wording,
pursuit attentions are altered, and gender inequality sustained.

This report is set to analyze how machine learning can help mitigate the discrimination that is projected
onto the applicant, by studying how natural language processing tasks, such as named entity recognition
and sentiment analysis, can censor demographic characteristics to emphasize experience and skills, and
additionally assists in gender neutralizing job applications and advertisements to enhance pursuit intentions. By splitting the problem up in two separate analyses that operate on separate domain-irrelevant
datasets, several supervised neural networks were composed based on LSTM, BiLSTM and CNN’s, to
perform binary and multi-label classification. Precision, recall and F1-score were measured to compare
all networks during testing. The results showed that NER can assist in recognizing named entities, such
as demographic characteristics, given the dataset have a uniformly and strongly represented amount of
named entities upon training. The results also showed that a binary and multi-label classification of text
regarding gender polarization were not efficient, with score measurements way less than required, among
other things due to the fact that the lingo of the dataset was influenced by its origin and the lacking
diversity of gender polarities.

The results ultimately suggest, that machine learning performing named entity recognition can assist
in censoring demographic characteristics if an adequate dataset is generated of sufficient size, and that
to perform sentiment analysis to classify gendered wording would require a much more diverse dataset
based on a more valid methodological approach regarding categorising gender-biased words.
