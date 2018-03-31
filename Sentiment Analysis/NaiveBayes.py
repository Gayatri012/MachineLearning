# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:17:55 2018

"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import Counter
import sys
from nltk import word_tokenize



class NaiveBayes:

    positive_counts = 0.0
    negative_counts = 0.0
    positive_review_count = 0.0
    negative_review_count = 0.0
    prob_positive = 0.0
    prob_negative = 0.0


    
    def trainData(self, fileName):
        print("Model training using train data")
        print("------------------------------------")
        
        # First we read data from the excel file in the dataset
        dataset = pd.read_csv(fileName, sep='\t', encoding='latin1')
        #print(len(dataset))
        #Combining train data and train target data into a list
        train_data = list(zip(dataset['Sentence'], dataset['Sentiment']))
        
        #Extracting all negative sentiment text from data
        negative_train_text = self.get_text(train_data, 0)
        #Extracting all positive sentiment text from data
        positive_train_text = self.get_text(train_data, 1)
        
        # Generate word counts for negative tone.
        self.negative_counts = self.count_text(negative_train_text)
        # Generate word counts for positive tone.
        self.positive_counts = self.count_text(positive_train_text)
        
        # We need these counts to use for smoothing when computing the prediction.
        self.positive_review_count = self.get_y_count(1, train_data)
        self.negative_review_count = self.get_y_count(0, train_data)
    
        # Class probabilities
        self.prob_positive = self.positive_review_count / len(train_data)
        self.prob_negative = self.negative_review_count / len(train_data)
    
        print("Probability of Positive Sentiment : ", self.prob_positive)
        print("Probability of Negative Sentiment : ", self.prob_negative)
        
        #Predictions based on the train data
        predictions = [self.make_decision(row[0], self.negative_counts, self.prob_negative, self.negative_review_count, self.positive_counts, self.prob_positive, self.positive_review_count)
        for row in train_data]
        
        # We check the accuracy by comparing with actual data
        actual = list(dataset['Sentiment'])
    
        accuracy = sum(1 for i in range(len(predictions)) if predictions[i] == actual[i]) / float(len(predictions))
    
        print("Accuracy of overall prediction of train data: " ,"{0:.4f}".format(accuracy))
        print("Confusion matrix of train data: ")
        print(confusion_matrix(dataset['Sentiment'], predictions))
        
    
    def predictData(self, fileName):
        print("\nPrediction result on test data")
        print("------------------------------------")
        dataset = pd.read_csv(fileName, sep='\t', encoding='latin1')
        
        #Creating a list of test data
        test_data = list(zip(dataset['Sentence'], dataset['Sentiment']))
        
        # We make predictions based on the model we have trained the training data with
        predictions = [self.make_decision(row[0], self.negative_counts, self.prob_negative, self.negative_review_count, self.positive_counts, self.prob_positive, self.positive_review_count)
        for row in test_data]
        
        actual = list(dataset['Sentiment'])
    
        accuracy = sum(1 for i in range(len(predictions)) if predictions[i] == actual[i]) / float(len(predictions))
    
        print("Accuracy of overall prediction of test data : " ,"{0:.4f}".format(accuracy))
        print("Confusion matrix of test data: ")    
        print(confusion_matrix(dataset['Sentiment'], predictions))
        
    
    # We need a function that will split the text based upon sentiment
    def get_text(self, reviews, score):
      # Join together the text in the Sentence for a particular sentiment.
      # We lowercase to avoid "Not" and "not" being seen as different words, for example.
       
        s = " "
        for row in reviews:
            if row[1] == score:
                s = s + " " + row[0].lower()
        return s
    
            
    
    # Count word frequency for each sentiment of text
    def count_text(self, text):
      # Split text into words using nltk's word_tokenize
      words = word_tokenize(text)
      # Count up the occurence of each word.
      return Counter(words)
    
    
    # We need this function to calculate a count of a given classification
    def get_y_count(self, score, train_data):
      # Compute the count of each classification occuring in the data.
        c = 0
        for row in train_data:
            if row[1] == score:
                c = c + 1
        
        return c
    
    
    # Finally, we create a function that will, given a text example, allow us to calculate the probability
    # of a positive, negative or NA review
    def make_class_prediction(self, text, counts, class_prob, class_count):
      prediction = 1
      text_counts = Counter(word_tokenize(text))
      
      if (sum(counts.values()) + class_count) != 0:
          for word in text_counts:
          # For every word in the text, we get the number of times that word occured in the reviews for a given class, add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count to also smooth the denominator).
          # Smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data.
          # We also smooth the denominator counts to keep things even.
              prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
      else:
            prediction = 0.0
          
      # Now we multiply by the probability of the class existing in the documents.
      return prediction * class_prob
    
    
    # Here we will create a function that will actually make the prediction
    def make_decision(self, text, negative_counts, prob_negative, negative_review_count, positive_counts, prob_positive, positive_review_count):
        # Compute the negative and positive probabilities.
        negative_prediction = self.make_class_prediction(text, negative_counts, prob_negative, negative_review_count)
        positive_prediction = self.make_class_prediction(text, positive_counts, prob_positive, positive_review_count)
    
        # We assign a classification based on which probability is greater.
        if negative_prediction > positive_prediction:
            return 0
        else:
            return 1
    
    

nbObj = NaiveBayes()

#Train the model using training data
nbObj.trainData(sys.argv[1])

#Predict test using model created already 
nbObj.predictData(sys.argv[2])    
