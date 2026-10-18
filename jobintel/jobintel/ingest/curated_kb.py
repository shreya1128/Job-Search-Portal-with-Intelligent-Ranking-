"""Hand-curated knowledge graph (the examples in the project brief, extended).  Source tag: 'curated'.
  HIERARCHY : skill -> sub-skills / frameworks   (relation broader_than)
  ROLES     : occupation -> essential / optional skills (requires_essential / requires_optional)
  SIMILAR   : occupation <-> occupation (related_to)
This is deliberately small and human-readable: it drives knowledge-based recommendation, related-skill discovery and
cold-start suggestions, and it is complemented by the much larger ESCO and O*NET graphs."""

HIERARCHY = {
    "python": ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "seaborn", "django", "flask", "fastapi", "tensorflow", "pytorch", "keras", "opencv", "nltk", "spacy"],
    "machine learning": ["supervised learning", "unsupervised learning", "reinforcement learning", "deep learning", "natural language processing", "computer vision",
                         "feature engineering", "regression", "classification", "clustering", "recommendation systems", "scikit-learn", "xgboost"],
    "deep learning": ["neural networks", "tensorflow", "pytorch", "keras", "computer vision", "natural language processing"],
    "data science": ["machine learning", "statistics", "data analysis", "data visualization", "data mining", "python", "r", "sql"],
    "data engineering": ["etl", "big data", "data warehousing", "spark", "hadoop", "kafka", "airflow", "sql", "data modeling"],
    "big data": ["hadoop", "spark", "hive", "hbase", "kafka", "flink"],
    "data visualization": ["tableau", "power bi", "matplotlib", "seaborn", "plotly", "excel"],
    "business intelligence": ["power bi", "tableau", "qlik", "sql", "data warehousing", "excel"],
    "sql": ["mysql", "postgresql", "sql server", "oracle", "sqlite", "pl/sql", "t-sql"],
    "nosql": ["mongodb", "cassandra", "redis", "dynamodb", "neo4j", "elasticsearch"],
    "aws": ["ec2", "s3", "lambda", "sagemaker", "redshift", "dynamodb"],
    "cloud computing": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform"],
    "devops": ["docker", "kubernetes", "jenkins", "ci/cd", "terraform", "ansible", "git", "linux"],
    "web development": ["html", "css", "javascript", "react", "angular", "vue", "node.js", "typescript"],
    "java": ["spring boot", "spring", "hibernate", "junit", "microservices"],
    "javascript": ["react", "angular", "vue", "node.js", "typescript"],
    "natural language processing": ["nltk", "spacy", "hugging face", "text mining", "large language models"],
    "cybersecurity": ["network security", "penetration testing", "siem", "cissp"],
    "software engineering": ["data structures", "algorithms", "design patterns", "system design", "unit testing", "agile", "git"],
    "agile": ["scrum", "kanban", "jira"],
    "accounting": ["bookkeeping", "accounts payable", "accounts receivable", "auditing", "quickbooks", "gaap", "ifrs", "tax preparation"],
    "financial analysis": ["financial modeling", "forecasting", "budgeting", "excel", "risk management"],
    "nursing": ["patient care", "medication administration", "electronic health records", "basic life support", "hipaa", "triage"],
}

ROLES = {   # role: (essential skills, optional skills)
    "Machine Learning Engineer": (["python", "machine learning", "sql", "tensorflow", "docker"], ["pytorch", "deep learning", "aws", "kubernetes", "scikit-learn", "git", "spark"]),
    "Data Scientist": (["python", "machine learning", "statistics", "sql", "data visualization"], ["pandas", "numpy", "scikit-learn", "tableau", "deep learning", "r", "big data"]),
    "Data Analyst": (["sql", "excel", "data analysis", "data visualization"], ["python", "tableau", "power bi", "pandas", "statistics", "r"]),
    "Data Engineer": (["python", "sql", "etl", "data warehousing", "spark"], ["airflow", "kafka", "aws", "hadoop", "docker", "snowflake", "big data"]),
    "AI Engineer": (["python", "machine learning", "deep learning", "tensorflow"], ["pytorch", "natural language processing", "computer vision", "docker", "large language models"]),
    "NLP Engineer": (["python", "natural language processing", "machine learning"], ["spacy", "nltk", "hugging face", "pytorch", "deep learning"]),
    "Computer Vision Engineer": (["python", "computer vision", "deep learning", "opencv"], ["tensorflow", "pytorch", "c++", "docker"]),
    "Business Intelligence Developer": (["sql", "business intelligence", "power bi", "data warehousing"], ["tableau", "etl", "excel", "python"]),
    "Software Engineer": (["data structures", "algorithms", "git", "object oriented programming"], ["java", "python", "sql", "agile", "docker", "system design"]),
    "Backend Developer": (["sql", "rest api", "git"], ["java", "python", "node.js", "spring boot", "django", "docker", "microservices", "redis"]),
    "Frontend Developer": (["javascript", "html", "css"], ["react", "angular", "vue", "typescript", "git"]),
    "Full Stack Developer": (["javascript", "html", "css", "sql", "rest api"], ["react", "node.js", "python", "docker", "mongodb", "git"]),
    "DevOps Engineer": (["linux", "docker", "ci/cd", "git"], ["kubernetes", "jenkins", "terraform", "ansible", "aws", "bash"]),
    "Cloud Engineer": (["aws", "linux", "docker"], ["azure", "gcp", "terraform", "kubernetes", "python", "devops"]),
    "Database Administrator": (["sql", "mysql", "data modeling"], ["oracle", "postgresql", "sql server", "linux", "nosql"]),
    "QA Engineer": (["unit testing", "agile"], ["selenium", "junit", "python", "java", "jira", "ci/cd"]),
    "Cybersecurity Analyst": (["cybersecurity", "network security", "linux"], ["siem", "penetration testing", "cissp", "python"]),
    "Financial Analyst": (["financial analysis", "excel", "financial modeling"], ["forecasting", "budgeting", "sql", "tableau", "accounting"]),
    "Accountant": (["accounting", "bookkeeping", "excel"], ["quickbooks", "accounts payable", "accounts receivable", "tax preparation", "gaap", "sap"]),
    "Registered Nurse": (["nursing", "patient care", "basic life support"], ["medication administration", "electronic health records", "advanced cardiac life support", "hipaa", "triage"]),
}

SIMILAR = [("Machine Learning Engineer", "Data Scientist"), ("Machine Learning Engineer", "AI Engineer"), ("Data Scientist", "Data Analyst"),
           ("Data Engineer", "Backend Developer"), ("Data Engineer", "Machine Learning Engineer"), ("NLP Engineer", "AI Engineer"),
           ("Computer Vision Engineer", "AI Engineer"), ("Software Engineer", "Backend Developer"), ("Full Stack Developer", "Frontend Developer"),
           ("Full Stack Developer", "Backend Developer"), ("DevOps Engineer", "Cloud Engineer"), ("Business Intelligence Developer", "Data Analyst"),
           ("Financial Analyst", "Accountant"), ("QA Engineer", "Software Engineer")]
