import pandas as pd
from pycaret.classification import*


#تحميل البيانات
data = pd.read_csv('Student_performance_data _.csv')


cld = setup(data, 
            target='GradeClass' ,
            session_id= 123 ,
            numeric_features = ['Age' , 'StudyTimeWeekly', 'Absences'],
            categorical_features = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music','Volunteering'],
            ignore_features=['StudentID', 'GPA'])

best_model = compare_models()  

save_model(best_model , 'student_performance_model') 

create_api(best_model, 'student_performance_api')

print("تم حفظ النموذج بنجاح")

														
