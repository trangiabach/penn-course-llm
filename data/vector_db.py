from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import json
import pinecone
import os

load_dotenv()

data = json.load(open('courses.json'))
documents = []

for course_code in data.keys():
    course = data[course_code]
    title = course['name']
    department = course['department']
    description = course['description']
    prereqs_text = ""

    if 'prereqs' in course:
        prereqs = course['prereqs']

        if 'OR' in prereqs:
            or_text_contents = []
            for prereq in prereqs['OR']:
                if prereq in data:
                    prereq_course = data[prereq]
                    prereqs_text += f"{prereq} - {prereq_course['name']}"
                    or_text_contents.append(prereqs_text)
                else:
                    or_text_contents.append(prereq)
            or_text = ' OR '.join(or_text_contents)
            prereqs_text += or_text
        if 'AND' in prereqs:
            and_text_contents = []
            for prereq in prereqs['AND']:
                if 'OR' in prereq:
                    and_text_contents.append(' OR '.join(prereq['OR']))
                elif prereq in data:
                    prereq_course = data[prereq]
                    prereqs_text += f"{prereq} - {prereq_course['name']}"
                    and_text_contents.append(prereqs_text)
                else:
                    and_text_contents.append(prereq)
            and_text = ' AND '.join(and_text_contents)
            prereqs_text += and_text

    text_content = f"""Information about {course_code} - {title}: {description}."""

    if prereqs_text:
        text_content += f"""The prerequiste for this course is as follows: ${prereqs_text}"""

    document = Document(page_content=text_content, metadata={
                        "text": text_content, "course_code": course_code, "course_title": title, "course_description": description})

    documents.append(document)

embeddings = OpenAIEmbeddings()

index_name = 'penn-courses'

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),
)

if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
    Pinecone.from_documents(documents, embeddings, index_name=index_name)

vectorstore = Pinecone.from_existing_index(index_name, embeddings)
