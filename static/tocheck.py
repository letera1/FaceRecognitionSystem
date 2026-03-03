import joblib

model = joblib.load('face_recognition_model.pkl')



print(type(model))  # See what kind of object it is
print(model)        # Print the contents (may be large!)
print(model.classes_)  # Shows names of people it recognizes
