from fastapi import FastAPI, Query
from pycaret.classification import load_model, predict_model
import pandas as pd

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

model = load_model("student_performance_api")

@app.get("/predict")
async def predict(
    Age: int = Query(..., description="عمر الطالب (من 15 إلى 18 سنة)"),
    Gender: int = Query(..., description="جنس الطالب (0: ذكر، 1: أنثى)"),
    Ethnicity: int = Query(..., description="العرق (0: قوقازي، 1: أفريقي أمريكي، 2: آسيوي، 3: آخر)"),
    ParentalEducation: int = Query(..., description="مستوى تعليم الوالدين (0: لا يوجد، 1: ثانوي، 2: بعض الدراسة الجامعية، 3: بكالوريوس، 4: أعلى)"),
    StudyTimeWeekly: float = Query(..., description="وقت الدراسة الأسبوعي (بالساعات، من 0 إلى 20)"),
    Absences: int = Query(..., description="عدد مرات الغياب خلال العام الدراسي (من 0 إلى 30)"),
    Tutoring: int = Query(..., description="الحصول على دروس خصوصية (0: لا، 1: نعم)"),
    ParentalSupport: int = Query(..., description="مستوى دعم الوالدين (0: لا يوجد، 1: منخفض، 2: متوسط، 3: عالي، 4: عالي جدًا)"),
    Extracurricular: int = Query(..., description="المشاركة في أنشطة لا منهجية (0: لا، 1: نعم)"),
    Sports: int = Query(..., description="المشاركة في الأنشطة الرياضية (0: لا، 1: نعم)"),
    Music: int = Query(..., description="المشاركة في الأنشطة الموسيقية (0: لا، 1: نعم)"),
    Volunteering: int = Query(..., description="المشاركة في العمل التطوعي (0: لا، 1: نعم)"),
):
    data = {
        "Age": Age,
        "Gender": Gender,
        "Ethnicity": Ethnicity,
        "ParentalEducation": ParentalEducation,
        "StudyTimeWeekly": StudyTimeWeekly,
        "Absences": Absences,
        "Tutoring": Tutoring,
        "ParentalSupport": ParentalSupport,
        "Extracurricular":Extracurricular, 
        "Sports": Sports, 
        "Music": Music ,
        "Volunteering": Volunteering

    }

    df = pd.DataFrame([data])

    predictions = predict_model(model, data=df)

    predicted_grade = predictions["prediction_label"].iloc[0]
    grade_map = {
        0: '(100-90 ممتاز)',
        1: '(100-90 جيد جدًا)',
        2: '(100-90 جيد)',
        3: '(100-90 مقبول)',
        4: '(100-90 راسب)'
    }

    grade = grade_map.get(predicted_grade, 'غير معروف')

    return {"الدرجة المتوقعة":grade}