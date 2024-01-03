import json

file = open('courses.json')

data = json.load(file)

print("Number of courses: " + str(len(data)))

departments = set()
total_text = ""

for course_code in data.keys():
    course = data[course_code]
    departments.add(course['department'])
    text_content = course['description'] + course['name']
    total_text += text_content

average_text_length = len(total_text) // len(data)

print("Number of course codes/departments: " + str(len(departments)))

print("Total text length: " + str(len(total_text)) + " characters")

print("Average text length: " + str(average_text_length) + " characters")

file.close()
