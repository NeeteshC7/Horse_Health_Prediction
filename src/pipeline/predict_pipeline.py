import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import pdb


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            print("inside predict")

            label_mapping = {0: 'Horse has died', 1: 'Horse was euthanized', 2: 'Horse will live'}
            
            predicted_label = label_mapping.get(preds[0], 'Unknown')
            
            return predicted_label
           # return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 surgery: float,
                 age: float,
                 hospital_number: float,
                 rectal_temp: float,
                 pulse: float,
                 respiratory_rate: float,
                 temp_of_extremities: float,
                 peripheral_pulse: float,
                 mucous_membrane: float,
                 capillary_refill_time: float,
                 pain: float,
                 peristalsis: float,
                 abdominal_distention: float,
                 nasogastric_tube: float,
                 nasogastric_reflux: float,
                 nasogastric_reflux_ph: float,
                 rectal_exam_feces: float,
                 abdomen: float,
                 packed_cell_volume: float,
                 total_protein: float,
                 abdomo_appearance: float,
                 abdomo_protein: float,
                 surgical_lesion: float,
                 lesion_1: float,
                 lesion_2: float,
                 cp_data: float):
        self.surgery = surgery
        self.age = age
        self.hospital_number = hospital_number
        self.rectal_temp = rectal_temp
        self.pulse = pulse
        self.respiratory_rate = respiratory_rate
        self.temp_of_extremities = temp_of_extremities
        self.peripheral_pulse = peripheral_pulse
        self.mucous_membrane = mucous_membrane
        self.capillary_refill_time = capillary_refill_time
        self.pain = pain
        self.peristalsis = peristalsis
        self.abdominal_distention = abdominal_distention
        self.nasogastric_tube = nasogastric_tube
        self.nasogastric_reflux = nasogastric_reflux
        self.nasogastric_reflux_ph = nasogastric_reflux_ph
        self.rectal_exam_feces = rectal_exam_feces
        self.abdomen = abdomen
        self.packed_cell_volume = packed_cell_volume
        self.total_protein = total_protein
        self.abdomo_appearance = abdomo_appearance
        self.abdomo_protein = abdomo_protein
        self.surgical_lesion = surgical_lesion
        self.lesion_1 = lesion_1
        self.lesion_2 = lesion_2
        self.cp_data = cp_data

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "surgery": [self.surgery],
                "age": [self.age],
                "hospital_number": [self.hospital_number],
                "rectal_temp": [self.rectal_temp],
                "pulse": [self.pulse],
                "respiratory_rate": [self.respiratory_rate],
                "temp_of_extremities": [self.temp_of_extremities],
                "peripheral_pulse": [self.peripheral_pulse],
                "mucous_membrane": [self.mucous_membrane],
                "capillary_refill_time": [self.capillary_refill_time],
                "pain": [self.pain],
                "peristalsis": [self.peristalsis],
                "abdominal_distention": [self.abdominal_distention],
                "nasogastric_tube": [self.nasogastric_tube],
                "nasogastric_reflux": [self.nasogastric_reflux],
                "nasogastric_reflux_ph": [self.nasogastric_reflux_ph],
                "rectal_exam_feces": [self.rectal_exam_feces],
                "abdomen": [self.abdomen],
                "packed_cell_volume": [self.packed_cell_volume],
                "total_protein": [self.total_protein],
                "abdomo_appearance": [self.abdomo_appearance],
                "abdomo_protein": [self.abdomo_protein],
                "surgical_lesion": [self.surgical_lesion],
                "lesion_1": [self.lesion_1],
                "lesion_2": [self.lesion_2],
                "cp_data": [self.cp_data],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
