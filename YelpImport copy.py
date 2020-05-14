#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:38:59 2019

@author: reubenallik
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gensim
import re
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


############################# DATA IMPORTS ##################################
json_files = ['business.json', 'checkin.json', 'review.json', 'tip.json', 'user.json']
variable_names = ['business', 'checkin', 'review', 'tip', 'user']


def loadBusiness():
    business = pd.read_json('business.json', lines = True)
#    for i in business:
#        i.to_csv('businesstest.csv')
#        break
    return business

def loadCheckin():
    checkin = pd.read_json('checkin.json', lines = True, chunksize=100)
#    for i in checkin:
#        i.to_csv('checkintest.csv')
#        break
    return checkin

def loadReview():
    review = pd.read_json('review.json', lines = True, chunksize=1000000)
    # final_review = pd.DataFrame([])
    # index = 0
    # pd.concat(final_review, [i for i in review])


    return review

def loadTip():
    tip = pd.read_json('tip.json', lines = True, chunksize=100)
#    for i in tip:
#        i.to_csv('tiptest.csv')
#        break
    return tip

def loadUser():
    user = pd.read_json('user.json', lines = True, chunksize=100)
#    for i in user:
#        i.to_csv('usertest.csv')
#        break
    return user



###############Loading Cities, now obsolite ##########
def loadCities(c):
    c.to_csv('cityCount.csv')
    return "done"


##############Loading categories and their frequencies ############

def loadBizCategories(business):
    categories = []
    for label, content in business.items():
        # print(type(content))
        if label=='categories':
            for index, value in content.items():
                try:
                    categoriesSplit = re.split(',', value)
                    for eachCategory in categoriesSplit:
                        categories.append(eachCategory.strip())
                    if index%10000==0:
                        print(index)
                except:
                    if index%10000==0:
                            print(index)
    return categories


def loadCategoryFreq(data, csvtitle):
    categories = loadBizCategories(data)
    catDF = pd.Series(categories)
    valueCounts = catDF.value_counts()
    valueCounts.to_csv(csvtitle, header=True)
    print('categories calculated, csv titled: ',csvtitle)
    return

################ Segmenting data sets by each type of business #########

def loadRestaurants(business):
    restaurant = []
    for label, content in business.items():
        if label=='categories':
            for index, value in content.items():
                try:
                    if 'Restaurants' in value and 'Home Services' not in value and 'Shopping' not in value and 'Beauty & Spas' not in value and 'Health & Medical' not in value:
                        restaurant.append(1)
                        if index%10000==0:
                            print(index)
                    else:
                        restaurant.append(0)
                except:
                    restaurant.append(0)
                    if index%10000==0:
                            print(index)
    business['IsRestaurant'] = restaurant
    restaurantDF = business.loc[business['IsRestaurant'] == 1]
    loadCategoryFreq(restaurantDF, 'categoryCountsRestaurants.csv')
    return restaurantDF

def loadShopping(business):
    shopping = []
    for label, content in business.items():
        # print(type(content))
        if label=='categories':
            for index, value in content.items():
                try:
                    if 'Shopping' in value and 'Home Services' not in value and 'Restaurants' not in value and 'Beauty & Spas' not in value and 'Health & Medical' not in value:
                        shopping.append(1)
                        if index%10000==0:
                            print(index)
                    else:
                        shopping.append(0)
                except:
                    shopping.append(0)
                    if index%10000==0:
                            print(index)
    business['IsShopping'] = shopping
    shoppingDF = business.loc[business['IsShopping'] == 1]
    loadCategoryFreq(shoppingDF, 'categoryCountsShopping.csv')
    return shoppingDF

def loadHomeServices(business):
    homeservices = []
    for label, content in business.items():
        if label=='categories':
            for index, value in content.items():
                try:
                    if 'Home Services' in value and 'Restaurants' not in value and 'Shopping' not in value and 'Beauty & Spas' not in value and 'Health & Medical' not in value:
                        homeservices.append(1)
                        if index%10000==0:
                            print(index)
                    else:
                        homeservices.append(0)
                except:
                    homeservices.append(0)
                    if index%10000==0:
                            print(index)
    business['IsHomeServices'] = homeservices
    homeservicesDF = business.loc[business['IsHomeServices'] == 1]
    loadCategoryFreq(homeservicesDF, 'categoryCountsHomeServices.csv')
    return homeservicesDF

def loadBeautySpas(business):
    beautyspas = []
    for label, content in business.items():
        if label=='categories':
            for index, value in content.items():
                try:
                    if 'Beauty & Spas' in value and 'Home Services' not in value and 'Restaurants' not in value and 'Shopping' not in value and 'Health & Medical' not in value:
                        beautyspas.append(1)
                        if index%10000==0:
                            print(index)
                    else:
                        beautyspas.append(0)
                except:
                    beautyspas.append(0)
                    if index%10000==0:
                            print(index)
    business['IsBeautySpas'] = beautyspas
    beautyspasDF = business.loc[business['IsBeautySpas'] == 1]
    loadCategoryFreq(beautyspasDF, 'categoryCountsBeautySpas.csv')
    return beautyspasDF

def loadHealthMedical(business):
    healthmedical = []
    for label, content in business.items():
        if label=='categories':
            for index, value in content.items():
                try:
                    if 'Health & Medical' in value and 'Beauty & Spas' not in value and 'Home Services' not in value and 'Restaurants' not in value and 'Shopping' not in value:
                        healthmedical.append(1)
                        if index%10000==0:
                            print(index)
                    else:
                        healthmedical.append(0)
                except:
                    healthmedical.append(0)
                    if index%10000==0:
                            print(index)
    business['IsHealthMedical'] = healthmedical
    healthmedicalDF = business.loc[business['IsHealthMedical'] == 1]
    loadCategoryFreq(healthmedicalDF, 'categoryCountsHealthMedical.csv')
    return healthmedicalDF

#################### Reviews Analysis #######################

def loadReviewDups(reviews):
    for i in reviews:
        i['multiReviewBiz'] = i.duplicated(['business_id'])
        print(i['multiReviewBiz'].sum())
    return reviews


def mergeBizAndReviews(bizTypeDF, reviews):
    bizTypeIDs = bizTypeDF['business_id']

    mergeDF = pd.DataFrame([])
    for chunk in reviews:
        mergeDF = pd.merge(bizTypeIDs, chunk, on='business_id', how='inner')
    return mergeDF


##### removing \n ##########

def removeCHR(CSV):
    data = pd.read_csv(CSV)
    reviewTXT = data['text']
    strippedTXT = reviewTXT.str.strip('\n().!?,')
    data['text'] = strippedTXT
    data.to_csv(CSV)
    return data


### tokenize ###

def tokenizeReviews(CSV):
    data = pd.read_csv(CSV)
    reviewTXT = data['text']
    tokenList = []
    for review in reviewTXT:
        try:
            token = nltk.word_tokenize(review)
            tokenList.append(token)
        except:
            print('didnt work here')
            tokenList.append([])

    data['tokens'] = tokenList
    data.to_csv(CSV)
    print('csv ', CSV, ' done')
    pass


################### Random Sample ######################

def randomSample(CSV, fivestar, fourstar, threestar, twostar, onestar, newCSV):
    data = pd.read_csv(CSV)

    five = data.loc[data['stars'] == 5]
    print('five = ', len(five))
    fiveSample = five.sample(n=fivestar, random_state=1)

    four = data.loc[data['stars'] == 4]
    print('four = ', len(four))
    fourSample = four.sample(n=fourstar, random_state=1)

    three = data.loc[data['stars'] == 3]
    print('three = ', len(three))
    threeSample = three.sample(n=threestar, random_state=1)

    two = data.loc[data['stars'] == 2]
    print('two = ', len(two))
    twoSample = two.sample(n=twostar, random_state=1)

    one = data.loc[data['stars'] == 1]
    print('one = ', len(one))
    oneSample = one.sample(n=onestar, random_state=1)

    concatDFs = [fourSample, threeSample, twoSample, oneSample]

    # finalSample = pd.concat(concatDFs, ignore_index=True)
    finalSample = fiveSample.append(concatDFs, ignore_index=True)


    finalSample.to_csv(newCSV)
    print(CSV, ' done')
    return finalSample


## word2vec

def wordToVec(inputData):
    data = inputData
    tokens = data['tokens']
    model = gensim.models.Word2Vec(tokens, size=100, window=5, min_count=5, workers=4)
    print('model built')
    return model


def wordToVecByStars(CSV, word):
    data = pd.read_csv(CSV, converters={'tokens': eval})

    five = data.loc[data['stars'] == 5]
    fiveModel = wordToVec(five)
    print(fiveModel.wv.most_similar(word))

    four = data.loc[data['stars'] == 4]
    fourModel = wordToVec(four)
    print(fourModel.wv.most_similar(word))

    three = data.loc[data['stars'] == 3]
    threeModel = wordToVec(three)
    print(threeModel.wv.most_similar(word))

    two = data.loc[data['stars'] == 2]
    twoModel = wordToVec(two)
    print(twoModel.wv.most_similar(word))

    one = data.loc[data['stars'] == 1]
    oneModel = wordToVec(one)
    print(oneModel.wv.most_similar(word))

    return


############## visulaization

def wordToVecByStarsVisualization(CSV, words, chartTitleCat):
    data = pd.read_csv(CSV, converters={'tokens': eval})
    chartTitle = 'Similar words for ' + chartTitleCat + ' Reviews'
    chartFileName = 'chart' + chartTitleCat

    five = data.loc[data['stars'] == 5]
    fiveModel = wordToVec(five)
    fiveChartTitle = chartTitle + ' - 5 Stars'
    fiveChartFileName = chartFileName + '5.png'
    wordToVecByStarsVisualizationGraphs(fiveModel, words, fiveChartTitle, fiveChartFileName)

    four = data.loc[data['stars'] == 4]
    fourModel = wordToVec(four)
    fourChartTitle = chartTitle + ' - 4 Stars'
    fourChartFileName = chartFileName + '4.png'
    wordToVecByStarsVisualizationGraphs(fourModel, words, fourChartTitle, fourChartFileName)


    three = data.loc[data['stars'] == 3]
    threeModel = wordToVec(three)
    threeChartTitle = chartTitle + ' - 3 Stars'
    threeChartFileName = chartFileName + '3.png'
    wordToVecByStarsVisualizationGraphs(threeModel, words, threeChartTitle, threeChartFileName)


    two = data.loc[data['stars'] == 2]
    twoModel = wordToVec(two)
    twoChartTitle = chartTitle + ' - 2 Stars'
    twoChartFileName = chartFileName + '2.png'
    wordToVecByStarsVisualizationGraphs(twoModel, words, twoChartTitle, twoChartFileName)


    one = data.loc[data['stars'] == 1]
    oneModel = wordToVec(one)
    oneChartTitle = chartTitle + ' - 1 Stars'
    oneChartFileName = chartFileName + '1.png'
    wordToVecByStarsVisualizationGraphs(oneModel, words, oneChartTitle, oneChartFileName)

    return

def wordToVecByStarsVisualizationGraphs(model, keys, chartTitle, chartFilename):
    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = []
        words = []
        for similar_word, _ in model.most_similar(word, topn=30):
            words.append(similar_word)
            embeddings.append(model[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

    tsne_plot_similar_words(chartTitle, keys, embeddings_en_2d, word_clusters, 0.7,
                            chartFilename)



def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()



######## Frequencies ################
def wordFreqDist(inputData):
    # data = pd.read_csv(CSV, converters={'tokens': eval})
    data = inputData
    tokens = data['tokens']
    tokenlist = []
    stop_words = set(stopwords.words('english'))
    stop_words.add(',')
    stop_words.add('!')
    stop_words.add('.')
    stop_words.add(')')
    stop_words.add('(')
    stop_words.add('?')
    stop_words.add('-')
    stop_words.add('...')
    stop_words.add('``')
    stop_words.add("''")
    stop_words.add('$')
    stop_words.add('s')
    stop_words.add("'s")
    stop_words.add("n't")
    for i in tokens:
        for word in i:
            lower = word.lower()
            if lower not in stop_words:
                tokenlist.append(lower)

    fdist = nltk.FreqDist(tokenlist)
    print(fdist.most_common(25))
    return(fdist.most_common(25))

########## Distribution by stars #################

def wordFreqByStars(CSV):
    top25 = []
    data = pd.read_csv(CSV, converters={'tokens': eval})

    five = data.loc[data['stars'] == 5]
    fiveTop25 = wordFreqDist(five)
    top25.append(fiveTop25)

    four = data.loc[data['stars'] == 4]
    fourTop25 = wordFreqDist(four)
    top25.append(fourTop25)

    three = data.loc[data['stars'] == 3]
    threeTop25 = wordFreqDist(three)
    top25.append(threeTop25)

    two = data.loc[data['stars'] == 2]
    twoTop25 = wordFreqDist(two)
    top25.append(twoTop25)

    one = data.loc[data['stars'] == 1]
    oneTop25 = wordFreqDist(one)
    top25.append(oneTop25)


    print(CSV, ' done')
    return top25




##################################################################
##################################################################
##################################################################
##################################################################
######################### Method Calls ###########################


# a = loadBusiness()
# b = loadCheckin()
# reviewData = loadReview()
# d = loadTip()
# e = loadUser()

# print(a.keys())

# cities = a['city'].value_counts()
# print(cities)
#
# print(a['categories'].value_counts())


# a = loadCities(cities)

################Checking each category and creating seperate data file ########################
# only_restaurants = loadRestaurants(a)
# only_restaurants.to_csv('onlyRestaurants.csv')
# only_homeservices = loadHomeServices(a)
# only_homeservices.to_csv('onlyHomeServices.csv')
# only_shopping = loadShopping(a)
# only_shopping.to_csv('onlyShopping.csv')
# only_beautyspas = loadBeautySpas(a)
# only_beautyspas.to_csv('onlyBeautySpas.csv')
# only_healthmedical = loadHealthMedical(a)
# only_healthmedical.to_csv('onlyHealthMedical.csv')

############## Merging biz and review data ######################

###   Restaurants
# restaurantCSV = pd.read_csv('onlyRestaurants.csv')
# # mergedRestaurantDF = mergeBizAndReviews(restaurantCSV, reviewData)
# # mergedRestaurantDF.to_csv('onlyRestaurantReviews.csv')

###   Shopping
# shoppingCSV = pd.read_csv('onlyShopping.csv')
# mergedShoppingDF = mergeBizAndReviews(shoppingCSV, reviewData)
# mergedShoppingDF.to_csv('onlyShoppingReviews.csv')

###   Home Services
# homeservicesCSV = pd.read_csv('onlyHomeServices.csv')
# mergedHomeServicesDF = mergeBizAndReviews(homeservicesCSV, reviewData)
# mergedHomeServicesDF.to_csv('onlyHomeServicesReviews.csv')

###   Beauty & Spas
# beautyspasCSV = pd.read_csv('onlyBeautySpas.csv')
# mergedBeautySpasDF = mergeBizAndReviews(beautyspasCSV, reviewData)
# mergedBeautySpasDF.to_csv('onlyBeautySpasReviews.csv')

### Health & Medical
# healthmedicalCSV = pd.read_csv('onlyHealthMedical.csv')
# HealthMedicalDF = mergeBizAndReviews(healthmedicalCSV, reviewData)
# HealthMedicalDF.to_csv('onlyHealthMedicalReviews.csv')

############## descriptive stats ######################

merged_csv = ['onlyRestaurantReviews.csv', 'onlyShoppingReviews.csv', 'onlyHomeServicesReviews.csv', 'onlyBeautySpasReviews.csv', 'onlyHealthMedicalReviews.csv']

# for type in merged_csv:
#  readIn = pd.read_csv(type)
#  print('for ',type, ' star counts = \n', readIn['stars'].value_counts())




samplingList = [['onlyRestaurantReviews.csv', 3888,	2592, 1342, 948, 1230, 'Sample_RestaurantReviews.csv'],
                ['onlyShoppingReviews.csv', 4802, 1830, 838, 644, 1885, 'Sample_ShoppingReviews.csv'],
                ['onlyHomeServicesReviews.csv', 5636, 539, 282, 423, 3120, 'Sample_HomeServicesReviews.csv'],
                ['onlyBeautySpasReviews.csv', 7359, 1381, 601, 660, 1827, 'Sample_BeautySpasReviews.csv'],
                ['onlyHealthMedicalReviews.csv', 8012, 921, 416, 651, 3660, 'Sample_HealthMedicalReviews.csv']]

# for category in samplingList:
#     samp = randomSample(category[0], category[1], category[2], category[3], category[4], category[5], category[6])

sampled_csv = ['Sample_RestaurantReviews.csv', 'Sample_ShoppingReviews.csv', 'Sample_HomeServicesReviews.csv', 'Sample_BeautySpasReviews.csv', 'Sample_HealthMedicalReviews.csv']

# wordFreqSave = []
# for cat in sampled_csv:
#     #removeCHR(cat)
#     #tokenizeReviews(cat)
#     #wordFreqDist(cat)
#     saveWF = wordFreqByStars(cat)
#     wordFreqSave.append([cat, saveWF])
#
# output = pd.DataFrame([wordFreqSave])
#
# output.to_csv('wordFreqbyCatbyStars.csv')


#stripping characters
# for type in merged_csv:
#     stripped = removeCHR(type)
#     print(stripped['text'].head())

#tokenizing
# for type in merged_csv:
#     tokenized = tokenizeReviews(type)



# frequency distributions

#dist = wordFreqDist('onlyHealthMedicalReviews.csv')




#word to vec
# wordTV = wordToVec('Sample_RestaurantReviews.csv')
# # print(wordTV.wv.most_similar("food"))

# word To Vec By Stars

#wtvByStars = wordToVecByStars('Sample_BeautySpasReviews.csv', 'staff')



############# Visualization ##############
sampled_csv = [['Sample_RestaurantReviews.csv', ['food', 'service', 'wait', 'staff'], 'Restaurants'],
               ['Sample_ShoppingReviews.csv', ['store', 'service', 'selection', 'staff'], 'Shopping'],
               ['Sample_HomeServicesReviews.csv', ['location', 'service', 'customer', 'staff'], 'HomeServices'],
               ['Sample_BeautySpasReviews.csv', ['job', 'service', 'beautiful', 'staff'], 'BeautySpas'],
               ['Sample_HealthMedicalReviews.csv', ['time', 'wait', 'advice', 'staff'], 'HealthMedical']]


for i in sampled_csv:
    wordToVecByStarsVisualization(i[0], i[1], i[2])


#missing values
# rvcsv = pd.read_csv('onlyRestaurantReviews.csv')
# missing = rvcsv.loc[rvcsv['tokens'] == '[]']
# print(missing['text'])
# rvcsv.drop(missing)
# rvcsv.to_csv('onlyRestaurantReviews.csv')
# bizReviewDups = loadReviewDups(c)

