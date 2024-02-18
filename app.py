import nltk
import re
import streamlit as st
import pickle


nltk.download('punkt')
nltk.download('stopwords')

# Loading Models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

# Cleaning Function
def cleanResume(txt):
    # Removing the links
    cleanTxt = re.sub('http\S+\s', ' ', txt)

    # Removing Emails
    cleanTxt = re.sub('@\S+', ' ', cleanTxt)

    # Removing hashtags
    cleanTxt = re.sub('#\S+', ' ', cleanTxt)

    # Removing RT and cc
    cleanTxt = re.sub('RT|cc', ' ', cleanTxt)

    # Removing special characters
    cleanTxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,./<=>?@[\]^_`{|}~"""), ' ', cleanTxt)

    # Removing extra characters
    cleanTxt = re.sub('r[^\x100-\x7f]', ' ', cleanTxt)

    # Removing programming sequences
    cleanTxt = re.sub('#\s', ' ', cleanTxt)

    return cleanTxt


# Web App
def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with latin-1
            resume_text = resume_bytes.decode(('latin-1'))

        cleaned_resume = cleanResume([resume_text])
        cleaned_resume = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(cleaned_resume)[0]

        # Mapping Category ID to Category Name
        category_mapping = {
            6: 'Data Science',
            12: 'HR',
            0: 'Advocate',
            1: 'Arts',
            24: 'Web Designing',
            16: 'Mechanical Engineer',
            22: 'Sales',
            14: 'Health and fitness',
            5: 'Civil Engineer',
            15: 'Java Developer',
            4: 'Business Analyst',
            21: 'SAP Developer',
            2: 'Automation Testing',
            11: 'Electrical Engineering',
            18: 'Operations Manager',
            20: 'Python Developer',
            8: 'DevOps Engineer',
            17: 'Network Security Engineer',
            19: 'PMO',
            7: 'Database',
            13: 'Hadoop',
            10: 'ETL Developer',
            9: 'DotNet Developer',
            3: 'Blockchain',
            23: 'Testing'
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)


# Python main
if __name__ == "__main__":
    main()

# --------------------- END ---------------------
