import json
from jobintel.ingest import common as C
from jobintel.ingest import loaders as L


def test_naukri_location_normalisation():
    cities, country = C.parse_naukri_location("Bangalore/Bengaluru, Hyderabad/Secunderabad, Mumbai (All Areas), Pune(Kharadi), Any Location")
    assert cities == ["Bengaluru", "Hyderabad", "Mumbai", "Pune"] and country is None


def test_experience_and_seniority_helpers():
    assert C.parse_exp_range("3-6") == (3, 6) and C.parse_exp_range("1 year  exp") == (1, 1) and C.parse_exp_range("5+") == (5, None)
    assert C.seniority_to_exp("Entry level") == (0, 2) and C.seniority_to_exp("Unknown thing") == (None, None)
    assert C.parse_numeric_range("35000-40000") == (35000, 40000) and C.parse_numeric_range("0-0") == (None, None) and C.parse_numeric_range("Jun-90") == (None, None)


def test_placeholder_companies_are_not_employers():
    assert C.is_placeholder_company('Company Undisclosed') and C.is_placeholder_company('CONFIDENTIAL') and C.is_placeholder_company(None)
    assert not C.is_placeholder_company('Accenture') and not C.is_placeholder_company('Unknown Labs Pvt Ltd')


def test_company_and_cluster():
    assert C.normalize_company("Tata Consultancy Services Pvt. Ltd.") == C.normalize_company("TATA consultancy services")
    assert C.normalize_company("Café Inc") == "cafe"
    assert C.cluster_for_title("Senior Data Scientist") == "data_ai" and C.cluster_for_title("Registered Nurse") == "healthcare"
    assert C.cluster_for_title("Head Chef") == "retail_hospitality" and C.cluster_for_title("Mechanical Engineer") == "engineering_manufacturing"
    assert C.cluster_for_title("Primary School Teacher") == "education" and C.cluster_for_title("Sales Executive") == "sales_marketing"
    assert C.cluster_for_title("Delivery Driver") == "logistics_transport" and C.cluster_for_title("Astronaut") == "other"
    assert C.cluster_for_title("Data Analyst, Sales Operations") == "data_ai"          # first matching cluster wins


def test_mcf_loader(tmp_path):
    jobs = [{"job_id": "Recruiter", "job_title": "Dev [$3k-$5k / Bukit]", "company_name": "A", "job_category": ["Admin ", " Secretarial, Marketing ", " Public Relations, Sales ", " Retail"],   # = "Admin / Secretarial, Marketing / Public Relations, Sales / Retail" split on '/' (as in the real file)
             "salary": "$4,000to$7,000Monthly", "min_experience": "3 years  exp", "skills_required": ["Python"], "requirements_and_role": "build things", "posting_date": "27 May 2019", "location": "X"},
            {"job_id": "Recruiter", "job_title": "Dev2", "company_name": "B", "job_category": [], "salary": "Salary undisclosed", "skills_required": [], "posting_date": "not_available"}]
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"jobs": jobs}))
    recs = list(L.load_mcf(p))
    assert recs[0].title == "Dev" and recs[0].category == "Admin / Secretarial" and recs[0].extras["all_categories"] == ["Admin / Secretarial", "Marketing / Public Relations", "Sales / Retail"] and recs[0].salary_min == 4000 and recs[0].salary_currency == "SGD"
    assert recs[0].experience_min == 3 and str(recs[0].posting_date) == "2019-05-27" and recs[1].salary_min is None and recs[1].posting_date is None
    assert recs[0].external_id != recs[1].external_id            # source ids are not unique -> disambiguated


def test_linkedin_tag_classification_and_loader(tmp_path):
    k = L.classify_linkedin_tag
    assert k("Python") == "skill" and k("High School Diploma") == "education" and k("Valid Driver's License") == "certification"
    assert k("3+ years of experience") == "experience" and k("Clearances (Act 151 Use code: 1KG6XN)") == "noise"
    p = tmp_path / "l.csv"
    p.write_text("job_link,job_title,company,job_location,first_seen,search_country,job_level,job_skills\nhttp://x/1,Data Analyst,ACME,\"San Diego, CA\",2024-01-12,United States,Associate,\"Python,SQL,Bachelor's degree\"\n")
    r = next(L.load_linkedin(p))
    assert r.city == "San Diego" and r.region == "CA" and r.text_quality == "skills_only" and r.education == "Bachelor's degree"


def test_emscad_loader(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("job_id,title,location,department,salary_range,company_profile,description,requirements,benefits,telecommuting,has_company_logo,has_questions,employment_type,required_experience,required_education,industry,function,fraudulent\n"
                 "7,Developer,\"US, CA, San Francisco\",,0-0,We are Acme.,Build web apps with Python for two years.,Bachelor's Degree,Free lunch,0,1,1,Full-time,Entry level,Bachelor's Degree,IT,Engineering,1\n")
    r = next(L.load_emscad(p))
    assert r.city == "San Francisco" and r.country == "United States" and r.salary_min is None and r.is_fraudulent
    assert r.company.startswith("Company-") and r.company_name_is_derived and (r.experience_min, r.experience_max) == (0, 2)
    assert "Requirements:\nBachelor" in r.description and r.category == "Engineering"
