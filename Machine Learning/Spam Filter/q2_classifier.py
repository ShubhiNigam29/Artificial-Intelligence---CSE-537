#Importing the libraries
import argparse
import csv
import numpy as np


#Parsing the content of command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f1', help='train dataset name', required=True)
parser.add_argument('-f2', help='test dataset name', required=True)
parser.add_argument('-o', help='output file name', required=True)

args = vars(parser.parse_args())

train = args['f1']
test = args['f2']
output_file = args['o']

#Initialization
def param_set():
    file = open(train,'r')
    non_spam = dict()
    spam_cnt = dict()
    limit = 8000
    spam_count = 0
    non_spam_count = 0
    count = 0
    alert_non_spam = False
    words = False
    
#Model to be trained
def spam_filter_naive_bayes(train, test, output_file): 
    
    limit = 8000
    smooth_param = 15
    # Creating dictionaries to store words and their frequencies in spam and non spam emails
    non_spam = dict()
    spam_cnt = dict()
    spam_count = 0
    non_spam_count = 0
    accuracy = 0
    rm = ","
    
    with open(train) as file:
          
        for count_line, r in enumerate(file,1):
            line = filter(lambda x: not (x in rm), r)
            count = 0
            alert_non_spam = False
            words = False
            for word in line.split():

                if count == 0:
                    count += 1
                    continue
                if count == 1:
                    if word == 'ham':
                        non_spam_count += 1
                        alert_non_spam = True
                    else:
                        spam_count += 1
                    count += 1
                    continue
                if words == False:
                    words = True
                    word_comp = word
                else:
                    words = False
                    # Checking if the word is a digit or the length of the word is too small, if yes: ignore
                    if word_comp.isdigit() or len(word_comp) < 4:
                        continue                
                    #Setting up the dictionaries
                    if alert_non_spam:
                        try:
                            if int(word) < limit:
                                non_spam[word_comp] += int(word)
                        except KeyError:
                            if int(word) < limit:
                                non_spam[word_comp] = int(word)
                    else:
                        try:
                            if int(word) < limit:
                                spam_cnt[word_comp] += int(word)
                        except KeyError:
                            if int(word) < limit:
                                spam_cnt[word_comp] = int(word)


    spam_store = dict()

    # Categorize the word if the frequency is within 1.25 standard deviations of the mean
    for key in spam_cnt.keys():
        if spam_cnt[key] < (np.mean(spam_cnt.values()) + np.std(spam_cnt.values()) * 1.25):
            spam_store[key] = spam_cnt[key]
    not_spam_word_count_new = dict()
    for key in non_spam.keys():
        if non_spam[key] < (np.mean(non_spam.values()) + np.std(non_spam.values()) * 1.25):
            not_spam_word_count_new[key] = non_spam[key]

    # Test
    
    file = open(test, 'r')
    count = 0
    ham_count = 0
    for line in file.readlines():

        if line.split(" ")[1] == 'ham':
            ham_count += 1
        count += 1

    row = 0

    file = open(test, 'r')
    out = open(output_file, 'wb')
    write_spam = csv.writer(out, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

 
    spam_pr = spam_count / float(spam_count + non_spam_count)
    for line in file.readlines():

        row += 1
        count = 0
        list_pr = []
        idx = ""
        numer = 1.0
        denom = 1.0
        for word in line.split(" "):

            if count == 0:
                idx = word
                count += 1
                continue
            if count == 1:
                y = word
                count += 1
                continue
            if count == 2:
                alert_word = False
                word_comp = word
                count += 1
                continue
            i=0
            c=0
            if count == 3:
                count += 1
            a = False
            while(alert_word == a):
                if(i == 2):
                    break
                try:
                    if word_comp.isdigit() or len(word_comp) < 4:
                        alert_word = True
                        continue
                    # Probability calculation
                    total_spam_pr = ((spam_pr * spam_store[word_comp]) + smooth_param) / float(
                            (spam_pr * spam_store[word_comp]) + (
                                (1 - spam_pr) * not_spam_word_count_new[word_comp] + smooth_param))
                    total_non_spam_pr = (non_spam[word_comp] + smooth_param) / float(
                            spam_cnt[word_comp] + non_spam[word_comp] + smooth_param)

                    # Consider if the probability is significant, 50%-50% spam non-spam will not give clear-cut idea about segregation
                    if total_spam_pr < 0.49 or total_spam_pr > 0.51:
                        list_pr.append(total_spam_pr)
                    c += 1
                except:
                    alert_word = True
                i += 1
                alert_word = True
                a = True
                
            if(c == 0):
                alert_word = False
                word_comp = word
                    
        for prob in list_pr:

            if prob == 0:
            
                continue

            # Checking if the total probability makes the email spam or ham
            numer *= prob
            denom *= (1 - prob)

        try:
            # Calculating total probability of email (for all the lines)
            total_pr = numer / float(numer + denom)
        except:

            continue

        if total_pr > 0.5:
            # Email is spam
            write_spam.writerow([idx, 'spam'])
            if y == 'spam':
                accuracy += 1

        else:
            # Email is ham
            write_spam.writerow([idx, 'ham'])
            if y == 'ham':
                accuracy += 1

    # Print accuracy
    print "Accuracy with smoothing: ", accuracy / float(row) * 100


# Calling the functions
spam_filter_naive_bayes(train, test, output_file)
