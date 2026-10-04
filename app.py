import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume & Job Description Matcher")

st.write(
    "Upload your resume and enter a job description "
    "to check how well your resume matches the job."
)

# Upload resume
resume_file = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)

# Enter job description
job_description = st.text_area(
    "Enter Job Description",
    height=250
)

# Analyze button
if st.button("Analyze Resume"):

    if resume_file is None:
        st.warning("Please upload your resume.")

    elif job_description.strip() == "":
        st.warning("Please enter the job description.")

    else:

        # Read resume
        reader = PdfReader(resume_file)

        resume_text = ""

        for page in reader.pages:
            text = page.extract_text()

            if text:
                resume_text += text

        # Compare resume and job description
        documents = [resume_text, job_description]

        vectorizer = TfidfVectorizer(stop_words="english")

        vectors = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        score = round(similarity * 100, 2)

        # Display score
        st.subheader("Resume Match Score")

        st.progress(min(score / 100, 1.0))

        st.success(
            f"Your resume matches this job description by {score}%"
        )

        # Matching keywords
        resume_words = set(resume_text.lower().split())
        job_words = set(job_description.lower().split())

        matched_words = resume_words.intersection(job_words)

        st.subheader("Matching Keywords")

        if matched_words:
            st.write(
                ", ".join(list(matched_words)[:30])
            )
        else:
            st.write("No matching keywords found.")

        # Missing keywords
        missing_words = job_words - resume_words

        st.subheader("Possible Missing Keywords")

        if missing_words:
            st.write(
                ", ".join(list(missing_words)[:30])
            )
        else:
            st.write("No major missing keywords found.")
