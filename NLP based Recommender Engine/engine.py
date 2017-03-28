import os
from requests import Session
import json
import pandas as pd
import codecs
from string import punctuation
from hebrew import stop_words
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, Normalizer
from pyspark.sql.types import StringType
from pyspark.mllib.stat import Statistics
import numpy as np
from pyspark.ml.clustering import LDA

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommenderEngine:


	def trainModel(self):
		
		logger.info("Training the model...")		

		query = '''select page_id, max(page_title) as page_title from cooladata where date_range(all) and page_id is not null group by page_id;'''

		def SQLtoURL(query):
    
    			data = query.replace('\n', ' ').replace('\t',' ').replace('   ',' ').replace('  ',' ')
    			return data


		def QueryCoola(query, file = None):
   
    			session = Session()
    			response = session.post(data = {'tq': query,}, url = 'https://app.cooladata.com/api/v2/projects/115659/cql/', headers = {'Authorization': 'Token dtQvPVejNcSebX1EkU0AqB2TJRXznIgZiDvDu3HR'},)
    			return response.content

		table = json.loads(codecs.decode(QueryCoola(SQLtoURL(query)),'utf-8'))['table']
		title_list = [x['c'] for x in table['rows']]
		table_cols = [d['label'] for d in table['cols']]  
		def convert_row(row):
    			rowlist = [d['v'] for d in row]
    			return rowlist

		rd = self.sc.parallelize(title_list).map(convert_row)
		titleData = self.spark.createDataFrame(rd, table_cols)
		titleData = titleData.dropna()
		
		hebrew_stopwords = stop_words()
		def rmv(words):
    			for punc in punctuation:
        			words = words.replace(punc,"")
    			for hword in hebrew_stopwords:
        			words = words.replace(hword, " ")
    			return words

		self.spark.udf.register("rmv", rmv, StringType())
		titleData.registerTempTable("wordstable")
		cleanedSentenceData = self.spark.sql("select page_id, page_title, rmv(page_title) as cleanedSentence from wordstable")
		tokenizer = Tokenizer(inputCol="cleanedSentence", outputCol="words")
		wordsData = tokenizer.transform(cleanedSentenceData)

		cv = CountVectorizer(inputCol="words", outputCol="rawFeatures", minDF = 2.0)
		cvModel = cv.fit(wordsData)
		featurizedData = cvModel.transform(wordsData)

		idf = IDF(inputCol="rawFeatures", outputCol="features")
		idfModel = idf.fit(featurizedData)
		rescaledData = idfModel.transform(featurizedData)

		lda = LDA(k=100)
		ldaModel = lda.fit(rescaledData)
		postFactorizedData = ldaModel.transform(rescaledData)

		norm = Normalizer(inputCol = "topicDistribution", outputCol="normTopicDist")
		scaledFactorizedNormalizedData = norm.transform(postFactorizedData)
		
		self.model = scaledFactorizedNormalizedData
		
		logger.info("model is built!")


	def __init__(self,spark,sqlctx,sc):
		
		logger.info("Starting up the Recommendation Engine: ")
		
		self.spark = spark
		self.sqlctx = sqlctx
		self.sc = sc
		self.trainModel()

	def recommend2user(self,user_id):
		
		query = '''select page_id from cooladata where date_range(last 21 days) and user_id = {:d} and page_id is not null group by page_id;'''.format(user_id)

		def SQLtoURL(query):
			data = query.replace('\n', ' ').replace('\t',' ').replace('   ',' ').replace('  ',' ')
			return data


		def QueryCoola(query, file = None):
			session = Session()
			response = session.post(data = {'tq': query,}, url = 'https://app.cooladata.com/api/v2/projects/115659/cql/', headers = {'Authorization': 'Token dtQvPVejNcSebX1EkU0AqB2TJRXznIgZiDvDu3HR'},)
			return response.content
		


		table = json.loads(codecs.decode(QueryCoola(SQLtoURL(query)),'utf-8'))['table']
		title_list = [x['c'] for x in table['rows']]
		table_cols = [d['label'] for d in table['cols']]  

		def convert_row(row):
			rowlist = []
			rowlist = [d['v'] for d in row]
			return rowlist

		rd = self.sc.parallelize(title_list).map(convert_row)
		historyTitleData = self.spark.createDataFrame(rd, table_cols)
		historyTitleData = historyTitleData.dropna()
		
		self.model.createOrReplaceTempView("Database")
		historyTitleData.registerTempTable("historyTable")
		
		pageVectorHistory = self.spark.sql('''select d.page_id, d.normTopicDist, case when h.page_id is null then 0 else 1 end as label from Database as d left join historyTable as h on d.page_id = h.page_id''')
		
		mainRdd = pageVectorHistory[pageVectorHistory['label'] == 1][['normTopicDist']].rdd.map(lambda x: x['normTopicDist'].toArray())
		mainVec = Statistics.colStats(mainRdd).mean()

		pageRank = pageVectorHistory[pageVectorHistory['label'] == 0].rdd.map(lambda row: (row['page_id'], float(np.dot(mainVec, row['normTopicDist'].toArray()))))
		pager = pageRank.toDF()
		pager.createOrReplaceTempView("pager")
		sortPageR = self.sqlctx.sql('''select _1 as page_id, _2 as similarity from pager order by similarity desc''')

		return sortPageR.take(10)

		