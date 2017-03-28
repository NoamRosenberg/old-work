
from flask import Flask, jsonify
import cherrypy
from paste.translogger import TransLogger
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from engine import RecommenderEngine

def init_spark():
	sc = SparkContext(pyFiles=['engine.py','hebrew.py'])
	spark = SparkSession.builder.appName("Recommendation-server").getOrCreate()
	sqlctx = SQLContext(sc)

	return spark, sqlctx, sc

spark, sqlctx, sc = init_spark()

recommender = RecommenderEngine(spark, sqlctx, sc)

app = Flask(__name__)
app.debug = True

@app.route("/train")
def train():
	recommender.trainModel()

@app.route("/getrecommendationfor/<int:user_id>", methods = ['GET'])
def getRecommendationFor(user_id):
	top_recommendations = recommender.recommend2user(user_id)
	return jsonify({'the users recommendation is':top_recommendations})


def run_server(app):
	
	app_logged = TransLogger(app)
	
	cherrypy.tree.graft(app_logged,'/')

	cherrypy.config.update({
		'engine.autoreload_on': True,
		'log.screen': True,
		'server.socket_port': 80,
		'server.socket_host':'0.0.0.0'
	})

	cherrypy.engine.start()
	cherrypy.engine.block()
	

if __name__=="__main__":
	
    	run_server(app)

