"""Pages that are in the navigation but not built yet.  They say so plainly (no fake content) and point at what works today.  Each later stage of Phase 7 replaces one of these with the
real page; the text below is the plan for it."""
from __future__ import annotations

from flask import render_template

PAGES = {
    "recommendations": dict(title="Recommendations", nav="recommend.recommendations", stage="Stage 3",
                            lead="Jobs picked for your skills, experience and preferences, each with a plain sentence on why it was picked.",
                            will=["Start from a resume or a short skills form, and edit what was understood before it is used.",
                                  "Each recommendation says which of your skills it uses and which it asks for that you do not list.",
                                  "Where there is too little history to personalise, the page says so instead of guessing."]),
    "resume": dict(title="Resume", nav="resume.analyzer", stage="Stage 3",
                   lead="Upload a resume and review the profile Jobintel reads from it: skills, experience, education and contact-free summary.",
                   will=["PDF, Word and plain-text resumes, with limits on size and pages and clear messages when a file cannot be read.",
                         "An editable list of the skills found, so you decide what the recommendations use.",
                         "Nothing from the file is kept unless you save the profile."]),
    "insights": dict(title="Career Insights", nav="market.market", stage="Stage 4",
                     lead="What the postings in the data ask for, which skills you are missing for a job you pick, and roles related to yours.",
                     will=["Skills in demand by job area, city and experience, counted from the stored postings, with the coverage of the data stated.",
                           "A skill-gap view for a job or role, ordered by how often the market asks for each skill.",
                           "Related roles from the occupation knowledge base. Salary figures only where the data supports them, and the limits said next to them."]),
    "skillgap": dict(title="Skill gap", nav="skillgap.skill_gap", stage="Stage 4",
                     lead="The skills a job asks for that your profile does not list, ordered by how often the market asks for them.",
                     will=["Pick a job from Find Jobs and compare it with your profile.", "Skills you already have are shown next to the ones you lack.",
                           "No courses are suggested: Jobintel has no course data, so it will not invent any."]),
}


def soon(key: str):
    return render_template("soon.html", **PAGES[key])
