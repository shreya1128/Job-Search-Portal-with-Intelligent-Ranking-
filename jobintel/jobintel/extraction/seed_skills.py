"""Curated seed skill vocabulary: canonical | type | aliases.  Dataset tags, O*NET and ESCO extend it
(scripts/build_lexicon.py).  Used to (1) normalise noisy skill tags and (2) drive the Aho-Corasick skill automaton."""

SEED_TEXT = """
python|technical|python3;py;python programming
java|technical|core java;java programming;j2ee
javascript|technical|js;java script;ecmascript
typescript|technical|ts
c|technical|c programming
c++|technical|cpp;c plus plus
c#|technical|csharp;c sharp
r|technical|r programming;r language
scala|technical|
go|technical|golang
rust|technical|
kotlin|technical|
swift|technical|
php|technical|
ruby|technical|ruby on rails;rails
matlab|technical|
sas|tool|sas programming;base sas
sql|technical|structured query language;sql programming
pl/sql|technical|plsql;pl sql
t-sql|technical|tsql;transact-sql
bash|technical|shell scripting;shell;bash scripting;unix shell
perl|technical|
julia|technical|
vba|technical|excel vba;macros
html|technical|html5
css|technical|css3
machine learning|technical|ml;machine-learning;machine learning algorithms
deep learning|technical|dl;deep-learning;deep neural networks
natural language processing|technical|nlp;text mining;text analytics
computer vision|technical|cv;image processing;image recognition
neural networks|technical|neural network;ann;artificial neural networks
supervised learning|technical|
unsupervised learning|technical|
reinforcement learning|technical|
artificial intelligence|technical|ai
generative ai|technical|genai;gen ai;generative artificial intelligence
large language models|technical|llm;llms;large language model
statistics|knowledge|statistical analysis;statistical modeling;statistical modelling;applied statistics
predictive modeling|technical|predictive modelling;predictive analytics
data mining|technical|
data analysis|technical|data analytics;analytics
data visualization|technical|data visualisation;data viz
data science|knowledge|
data engineering|technical|
data modeling|technical|data modelling
data warehousing|technical|data warehouse;dwh;data warehouse concepts
etl|technical|etl development;extract transform load;etl tools
big data|technical|big data analytics;bigdata
feature engineering|technical|
time series|technical|time series analysis;time series forecasting;forecasting models
a/b testing|technical|ab testing;split testing
recommendation systems|technical|recommender systems;recommendation engines
regression|technical|linear regression;logistic regression
classification|technical|
clustering|technical|k-means;kmeans
business intelligence|technical|bi;bi tools
pandas|technical|
numpy|technical|
scipy|technical|
scikit-learn|technical|sklearn;scikit learn
tensorflow|technical|tensor flow;tf
keras|technical|
pytorch|technical|torch
opencv|technical|open cv
nltk|technical|
spacy|technical|
hugging face|technical|huggingface;transformers
langchain|technical|
xgboost|technical|
lightgbm|technical|
matplotlib|technical|
seaborn|technical|
plotly|technical|
spark|technical|apache spark;pyspark;spark sql
hadoop|technical|apache hadoop;hdfs;mapreduce;map reduce
hive|technical|apache hive;hiveql
pig|technical|apache pig
hbase|technical|
kafka|technical|apache kafka
airflow|technical|apache airflow
flink|technical|apache flink
dbt|technical|data build tool
sqoop|technical|
tableau|tool|tableau desktop
power bi|tool|powerbi;microsoft power bi;power-bi
excel|tool|microsoft excel;ms excel;advanced excel;ms-excel
looker|tool|
qlik|tool|qlikview;qlik sense
spss|tool|
jupyter|tool|jupyter notebook;jupyter notebooks
git|tool|version control;git version control
github|tool|
gitlab|tool|
jira|tool|
confluence|tool|
postman|tool|
mysql|technical|my sql
postgresql|technical|postgres;postgre sql
mongodb|technical|mongo db;mongo
oracle|technical|oracle database;oracle db
sql server|technical|mssql;ms sql;microsoft sql server;ms sql server
sqlite|technical|
redis|technical|
cassandra|technical|apache cassandra
elasticsearch|technical|elastic search;elk
snowflake|technical|
redshift|technical|amazon redshift
bigquery|technical|big query;google bigquery
dynamodb|technical|
neo4j|technical|
nosql|technical|no sql
aws|technical|amazon web services;aws cloud
azure|technical|microsoft azure;azure cloud
gcp|technical|google cloud;google cloud platform
docker|technical|containerization;containers
kubernetes|technical|k8s
terraform|technical|
ansible|technical|
jenkins|technical|
ci/cd|technical|cicd;continuous integration;continuous delivery;ci cd pipelines
linux|technical|unix;linux administration
devops|technical|
microservices|technical|micro services
rest api|technical|restful api;rest apis;restful apis;rest;web services
graphql|technical|
databricks|tool|
sagemaker|tool|aws sagemaker
s3|tool|amazon s3
ec2|tool|
lambda|tool|aws lambda
react|technical|reactjs;react.js;react js
angular|technical|angularjs
vue|technical|vue.js;vuejs
node.js|technical|nodejs;node js;node
django|technical|
flask|technical|
fastapi|technical|
spring boot|technical|springboot
spring|technical|spring framework
.net|technical|dotnet;asp.net;dot net;.net core
hibernate|technical|
android|technical|android development
ios|technical|ios development
flutter|technical|
selenium|tool|selenium webdriver
junit|tool|
agile|knowledge|agile methodology;agile methodologies;agile development
scrum|knowledge|
kanban|knowledge|
object oriented programming|technical|oop;oops;object-oriented programming;object oriented design
data structures|technical|data structures and algorithms;dsa
algorithms|technical|
design patterns|technical|
system design|technical|software architecture
unit testing|technical|test driven development;tdd
cybersecurity|technical|cyber security;information security;infosec
network security|technical|
penetration testing|technical|pen testing;ethical hacking
siem|tool|splunk
accounting|knowledge|
financial analysis|knowledge|financial analyst
financial modeling|knowledge|financial modelling
budgeting|knowledge|budget management
forecasting|knowledge|
auditing|knowledge|internal audit;audit
accounts payable|knowledge|
accounts receivable|knowledge|
bookkeeping|knowledge|
quickbooks|tool|
sap|tool|sap erp;sap fico
tax preparation|knowledge|taxation;tax
risk management|knowledge|risk assessment
gaap|knowledge|us gaap
ifrs|knowledge|
patient care|knowledge|
nursing|knowledge|
electronic health records|tool|ehr;emr;electronic medical records;epic
hipaa|knowledge|
medication administration|knowledge|
phlebotomy|knowledge|
clinical documentation|knowledge|
triage|knowledge|
basic life support|certification|bls
advanced cardiac life support|certification|acls
cpr|certification|cardiopulmonary resuscitation
project management|knowledge|project manager
pmp|certification|project management professional
cissp|certification|
ccna|certification|
communication|soft|communication skills;verbal communication;written communication;excellent communication
leadership|soft|leadership skills;team leadership
teamwork|soft|team work;team player;collaboration;collaborative
problem solving|soft|problem-solving;problem solving skills
critical thinking|soft|
time management|soft|
stakeholder management|soft|
negotiation|soft|
customer service|soft|customer support;client service
presentation skills|soft|presentation;public speaking
attention to detail|soft|detail oriented;detail-oriented
analytical skills|soft|analytical thinking;analytical;analytical ability
adaptability|soft|flexibility
creativity|soft|
"""

# Generic soft/business single words that must not be matched in free text on their own.
GENERIC_SINGLE_WORDS = {
    "management", "design", "sales", "research", "development", "marketing", "operations", "support", "analysis",
    "planning", "training", "english", "computer", "technology", "engineering", "science", "business", "service",
    "services", "finance", "quality", "security", "testing", "software", "programming", "database", "analytics",
    "it skills", "analytical", "computer science", "skills", "other", "recruitment", "administration",
}
# Tags that carry no skill information at all.
TAG_STOPLIST = {"it skills", "skills", "other", "others", "n/a", "na", "none", "null", "", "good", "excellent", "required",
                "bachelor", "degree", "experience", "knowledge", "ability", "strong", "years", "year", "computer science"}

# Ordinary English words that ESCO / O*NET list as "skills" (or that are brand names used as verbs/nouns).  They stay in the
# lexicon so dataset TAGS still normalise, but they are never matched inside running text.
FREE_TEXT_BLOCKLIST = {
    "absorb", "assembly", "call", "less", "productivity", "purchasing", "reporting", "sensors", "source", "statistical", "payroll",
    "electronics", "telecommunication", "digitization", "cybernetics", "crowdfunding", "supercomputing", "google", "facebook", "tiktok",
    "rage", "salt", "shiva", "staf", "dids", "emass", "ioms", "jsss", "lams",
}
