"""
ER_MAP/envs/disease_db.py
=========================
Comprehensive disease database: 50 diseases across 10 clinical classes.
Each disease has entries in 4 databases: DISEASES, VITALS, LABS, SOAP_HISTORY.
"""

# ============================================================================
# DISEASES DATABASE
# ============================================================================
DISEASES_DB = {}
VITALS_DB = {}
LAB_RESULTS_DB = {}
SOAP_HISTORY_DB = {}

# ---------------------------------------------------------------------------
# CLASS 1: CARDIOVASCULAR (5 diseases)
# ---------------------------------------------------------------------------

DISEASES_DB["Acute Myocardial Infarction"] = {
    "true_disease": "Acute Myocardial Infarction",
    "true_symptoms": ["crushing substernal chest pain", "diaphoresis", "left arm pain", "nausea", "shortness of breath"],
    "correct_treatment": "aspirin 325mg, heparin drip, nitroglycerin, emergent PCI, morphine for pain",
    "lethal_treatments": ["thrombolytics if hemorrhagic stroke suspected", "beta-blockers if cardiogenic shock"],
    "medical_history": "HTN, hyperlipidemia, smoker, family history of CAD",
    "difficulty": "medium",
    "critical_labs": ["troponin", "ECG", "BNP", "CK"],
}
VITALS_DB["Acute Myocardial Infarction"] = "HR 105, BP 90/60, RR 22, SpO2 94%, Temp 37.0C — tachycardic, hypotensive"
LAB_RESULTS_DB["Acute Myocardial Infarction"] = {
    "troponin": "Troponin I: 4.2 ng/mL (CRITICAL HIGH — consistent with acute MI)",
    "ECG": "ST elevation in leads II, III, aVF — inferior STEMI",
    "BNP": "BNP: 850 pg/mL (elevated — heart failure)",
    "CK": "CK-MB: 45 U/L (elevated — myocardial injury)",
    "CBC": "WBC 11.2, Hgb 14.0, Plt 220 — mild leukocytosis",
    "BMP": "Na 140, K 4.2, Cr 1.0, Glucose 180 — stress hyperglycemia",
}
SOAP_HISTORY_DB["Acute Myocardial Infarction"] = {
    "HPI": "62M presents with sudden onset crushing substernal chest pressure radiating to left arm and jaw, started 45 minutes ago while climbing stairs. Associated diaphoresis and nausea. Rates pain 9/10.",
    "ROS": {"CV": "chest pain, palpitations", "Resp": "mild dyspnea", "GI": "nausea, no vomiting", "Neuro": "no focal deficits", "MSK": "left arm pain"},
    "Past_Medical_History": "Hypertension x 10 years, Hyperlipidemia, Type 2 DM, Former smoker 30 pack-years quit 2 years ago",
    "Medications": "Metformin 1000mg BID, Lisinopril 20mg daily, Atorvastatin 40mg daily",
    "Allergies": "NKDA",
    "Social_History": "Retired construction worker, former smoker, occasional alcohol, lives with wife",
    "Physical_Examination": "Diaphoretic, clutching chest. JVD noted. S3 gallop on auscultation. Lungs with bibasilar crackles. Abdomen soft.",
}

DISEASES_DB["Aortic Dissection"] = {
    "true_disease": "Aortic Dissection",
    "true_symptoms": ["sudden tearing chest pain radiating to back", "blood pressure differential between arms", "severe diaphoresis", "feeling of impending doom", "pain worst at onset"],
    "correct_treatment": "IV esmolol for heart rate and blood pressure control, CT angiography chest, emergency cardiothoracic surgery consult, target HR below 60",
    "lethal_treatments": ["aspirin", "thrombolytics", "heparin", "anticoagulation"],
    "medical_history": "Uncontrolled HTN, Marfan syndrome",
    "difficulty": "hard",
    "critical_labs": ["CT_angio", "ECG", "D-dimer", "CXR"],
}
VITALS_DB["Aortic Dissection"] = "HR 110, BP R arm 190/110 L arm 130/75, RR 26, SpO2 95%, Temp 37.1C — BP differential between arms!"
LAB_RESULTS_DB["Aortic Dissection"] = {
    "troponin": "Troponin I: 0.08 ng/mL (borderline — may mimic MI)",
    "ECG": "Sinus tachycardia, no ST changes — rules out primary cardiac event",
    "D-dimer": "D-dimer: >5000 ng/mL (markedly elevated)",
    "CT_angio": "CT Angio: Type A dissection with intimal flap from ascending aorta to iliac bifurcation",
    "CXR": "Widened mediastinum on CXR",
    "CBC": "WBC 13.5, Hgb 11.2, Plt 180 — mild anemia from hemorrhage into false lumen",
}
SOAP_HISTORY_DB["Aortic Dissection"] = {
    "HPI": "48M presents with sudden onset severe tearing chest pain radiating to back between shoulder blades. Pain was maximal at onset. Associated diaphoresis and sense of impending doom. Rates pain 10/10.",
    "ROS": {"CV": "tearing chest pain, palpitations", "Resp": "mild dyspnea", "Neuro": "transient left leg weakness", "GI": "nausea", "MSK": "back pain"},
    "Past_Medical_History": "Hypertension x 15 years poorly controlled, Marfan habitus noted on prior visits",
    "Medications": "Amlodipine 10mg daily (poor compliance reported)",
    "Allergies": "NKDA",
    "Social_History": "Accountant, non-smoker, social drinker, cocaine use remote history",
    "Physical_Examination": "Tall, thin habitus. Aortic regurgitation murmur. BP differential R>L arm by 60mmHg. Diminished left femoral pulse. Neuro exam normal.",
}

DISEASES_DB["Cardiac Tamponade"] = {
    "true_disease": "Cardiac Tamponade",
    "true_symptoms": ["chest pressure", "muffled heart sounds", "jugular venous distension", "hypotension", "dyspnea"],
    "correct_treatment": "emergent pericardiocentesis, IV fluid bolus, bedside echocardiography, cardiothoracic surgery consult",
    "lethal_treatments": ["diuretics", "nitroglycerin", "beta-blockers"],
    "medical_history": "Recent pericarditis, malignancy",
    "difficulty": "hard",
    "critical_labs": ["echo", "ECG", "CXR"],
}
VITALS_DB["Cardiac Tamponade"] = "HR 125, BP 78/60, RR 28, SpO2 92%, Temp 37.4C — Beck's triad: hypotension, JVD, muffled heart sounds"
LAB_RESULTS_DB["Cardiac Tamponade"] = {
    "ECG": "Low voltage QRS, electrical alternans — classic tamponade",
    "echo": "Large pericardial effusion with right ventricular diastolic collapse — tamponade physiology",
    "CXR": "Enlarged cardiac silhouette, water-bottle heart",
    "troponin": "Troponin I: 0.15 ng/mL (mildly elevated)",
    "CBC": "WBC 9.0, Hgb 10.5, Plt 200 — mild anemia",
    "BMP": "Na 138, K 4.5, Cr 1.2, BUN 22",
}
SOAP_HISTORY_DB["Cardiac Tamponade"] = {
    "HPI": "58F presents with progressive chest pressure and shortness of breath over 3 days. Worsening dyspnea on exertion. Feels lightheaded when standing. Had viral illness 2 weeks ago.",
    "ROS": {"CV": "chest pressure, lightheadedness", "Resp": "progressive dyspnea", "GI": "no complaints", "Neuro": "dizziness on standing"},
    "Past_Medical_History": "Breast cancer in remission x 2 years, viral pericarditis 2 weeks ago",
    "Medications": "Tamoxifen 20mg daily, Ibuprofen 400mg TID for pericarditis",
    "Allergies": "Sulfa drugs",
    "Social_History": "School teacher, non-smoker, no alcohol",
    "Physical_Examination": "Anxious, sitting upright. JVD to angle of jaw. Heart sounds distant/muffled. Pulsus paradoxus 18mmHg. Lungs clear.",
}

DISEASES_DB["Atrial Fibrillation with RVR"] = {
    "true_disease": "Atrial Fibrillation with RVR",
    "true_symptoms": ["rapid irregular heartbeat", "palpitations", "lightheadedness", "chest discomfort", "dyspnea on exertion"],
    "correct_treatment": "IV diltiazem or metoprolol for rate control, anticoagulation with heparin, assess for cardioversion, echocardiogram",
    "lethal_treatments": ["cardioversion without anticoagulation if AFib duration unknown"],
    "medical_history": "HTN, prior AFib episodes",
    "difficulty": "medium",
    "critical_labs": ["ECG", "troponin", "TSH", "BMP"],
}
VITALS_DB["Atrial Fibrillation with RVR"] = "HR 155 irregular, BP 100/65, RR 22, SpO2 96%, Temp 37.0C — irregularly irregular rhythm"
LAB_RESULTS_DB["Atrial Fibrillation with RVR"] = {
    "ECG": "Atrial fibrillation with rapid ventricular response, rate 155, no ST changes",
    "troponin": "Troponin I: 0.04 ng/mL (normal — demand ischemia unlikely)",
    "TSH": "TSH: 0.1 mIU/L (LOW — consider hyperthyroidism as trigger)",
    "BMP": "Na 136, K 3.2 (LOW), Cr 0.9, Mg 1.5 (LOW)",
    "CBC": "WBC 7.8, Hgb 13.5, Plt 210 — normal",
    "BNP": "BNP: 420 pg/mL (elevated — atrial stretch)",
}
SOAP_HISTORY_DB["Atrial Fibrillation with RVR"] = {
    "HPI": "72M presents with 6 hours of rapid heartbeat, palpitations, and lightheadedness. Noticed irregular pulse at home. Mild chest discomfort. Denies syncope.",
    "ROS": {"CV": "palpitations, chest discomfort", "Resp": "dyspnea on exertion", "Neuro": "lightheadedness, no syncope", "GI": "no complaints"},
    "Past_Medical_History": "Hypertension, prior paroxysmal AFib (not on anticoagulation), Hyperthyroidism treated 5 years ago",
    "Medications": "Lisinopril 10mg daily, Aspirin 81mg daily",
    "Allergies": "NKDA",
    "Social_History": "Retired, 2 glasses wine daily, non-smoker",
    "Physical_Examination": "Alert, irregularly irregular pulse. No JVD. Lungs clear. No edema. Thyroid mildly enlarged.",
}

DISEASES_DB["Hypertensive Emergency"] = {
    "true_disease": "Hypertensive Emergency",
    "true_symptoms": ["severe headache", "blurred vision", "chest pain", "confusion", "nausea"],
    "correct_treatment": "IV nicardipine or nitroprusside drip, reduce BP by 25 percent in first hour, ICU admission, target organ damage workup",
    "lethal_treatments": ["rapid BP reduction to normal", "oral medications only"],
    "medical_history": "Chronic HTN, medication non-compliance",
    "difficulty": "medium",
    "critical_labs": ["BMP", "urinalysis", "ECG", "CT_head"],
}
VITALS_DB["Hypertensive Emergency"] = "HR 95, BP 240/140, RR 20, SpO2 97%, Temp 37.0C — severely hypertensive with end-organ symptoms"
LAB_RESULTS_DB["Hypertensive Emergency"] = {
    "BMP": "Na 138, K 3.8, Cr 2.1 (elevated — acute kidney injury), BUN 35",
    "urinalysis": "Urinalysis: proteinuria 3+, RBC casts — hypertensive nephropathy",
    "ECG": "LVH with strain pattern, no acute ST changes",
    "CT_head": "CT Head: no acute intracranial hemorrhage",
    "CBC": "WBC 8.5, Hgb 12.0, Plt 180, schistocytes on smear",
    "troponin": "Troponin I: 0.12 ng/mL (mildly elevated — demand ischemia)",
}
SOAP_HISTORY_DB["Hypertensive Emergency"] = {
    "HPI": "55M presents with worst headache of life, blurred vision, and confusion for 2 hours. Wife reports he ran out of BP medications 1 week ago. Mild chest discomfort.",
    "ROS": {"Neuro": "headache, confusion, blurred vision", "CV": "chest discomfort", "GI": "nausea", "Renal": "decreased urine output today"},
    "Past_Medical_History": "Hypertension x 20 years, CKD stage 3, non-compliant with medications",
    "Medications": "Prescribed: Amlodipine 10mg, Losartan 100mg, HCTZ 25mg — stopped all 1 week ago",
    "Allergies": "ACE inhibitors (angioedema)",
    "Social_History": "Construction worker, smokes 1 pack/day, drinks 6 beers on weekends",
    "Physical_Examination": "Confused, oriented x1. Papilledema on fundoscopy. S4 gallop. Lungs with crackles bilaterally. 1+ pedal edema.",
}

# ---------------------------------------------------------------------------
# CLASS 2: PULMONARY (5 diseases)
# ---------------------------------------------------------------------------

DISEASES_DB["Pulmonary Embolism"] = {
    "true_disease": "Pulmonary Embolism",
    "true_symptoms": ["sudden onset pleuritic chest pain", "dyspnea", "tachycardia", "hemoptysis", "unilateral leg swelling"],
    "correct_treatment": "anticoagulation with heparin, CT pulmonary angiography, consider thrombolytics if massive PE with hemodynamic instability",
    "lethal_treatments": ["thrombolytics if recent surgery or active bleeding"],
    "medical_history": "Recent surgery, OCP use, immobilization",
    "difficulty": "medium",
    "critical_labs": ["D-dimer", "CT_angio", "ECG", "ABG"],
}
VITALS_DB["Pulmonary Embolism"] = "HR 118, BP 95/60, RR 28, SpO2 89%, Temp 37.3C — tachycardic, hypoxic, tachypneic"
LAB_RESULTS_DB["Pulmonary Embolism"] = {
    "D-dimer": "D-dimer: 4200 ng/mL (markedly elevated)",
    "CT_angio": "CT Pulmonary Angiography: bilateral pulmonary emboli in segmental and subsegmental arteries, right heart strain",
    "ECG": "Sinus tachycardia, S1Q3T3 pattern, right axis deviation",
    "ABG": "pH 7.48, pCO2 28, pO2 62, HCO3 22 — respiratory alkalosis with hypoxemia",
    "troponin": "Troponin I: 0.25 ng/mL (elevated — right heart strain)",
    "CBC": "WBC 9.8, Hgb 12.1, Plt 195 — unremarkable",
    "BNP": "BNP: 580 pg/mL (elevated — RV strain)",
}
SOAP_HISTORY_DB["Pulmonary Embolism"] = {
    "HPI": "32F presents with sudden onset right-sided pleuritic chest pain and shortness of breath that started 3 hours ago at rest. Reports small amount of blood-tinged sputum. Had right hip replacement 2 weeks ago.",
    "ROS": {"Resp": "dyspnea, hemoptysis, pleuritic chest pain", "CV": "palpitations", "MSK": "right calf pain and swelling x 3 days", "Neuro": "no focal deficits"},
    "Past_Medical_History": "Right hip replacement 2 weeks ago, oral contraceptive use x 5 years, obesity BMI 34",
    "Medications": "Oral contraceptive pill, acetaminophen PRN, enoxaparin 40mg daily (stopped 3 days ago)",
    "Allergies": "NKDA",
    "Social_History": "Office worker, sedentary, non-smoker, no alcohol, no drug use",
    "Physical_Examination": "Anxious, tachypneic. Right calf swollen and tender with positive Homan sign. Lungs with decreased breath sounds right base. Heart tachycardic, regular.",
}

DISEASES_DB["Tension Pneumothorax"] = {
    "true_disease": "Tension Pneumothorax",
    "true_symptoms": ["acute chest pain", "severe dyspnea", "absent breath sounds unilateral", "tracheal deviation", "hypotension"],
    "correct_treatment": "emergent needle decompression 2nd intercostal space midclavicular line, followed by chest tube thoracostomy",
    "lethal_treatments": ["chest CT before decompression", "waiting for imaging"],
    "medical_history": "Tall thin habitus, COPD, recent trauma",
    "difficulty": "hard",
    "critical_labs": ["CXR", "ABG"],
}
VITALS_DB["Tension Pneumothorax"] = "HR 135, BP 70/40, RR 36, SpO2 82%, Temp 37.0C — hemodynamically unstable, hypoxic"
LAB_RESULTS_DB["Tension Pneumothorax"] = {
    "CXR": "Large right pneumothorax with mediastinal shift to left, tracheal deviation — TENSION PNEUMOTHORAX",
    "ABG": "pH 7.28, pCO2 55, pO2 48, HCO3 24 — respiratory acidosis with severe hypoxemia",
    "CBC": "WBC 10.0, Hgb 14.5, Plt 230 — normal",
    "BMP": "Na 140, K 4.0, Cr 0.8 — normal",
}
SOAP_HISTORY_DB["Tension Pneumothorax"] = {
    "HPI": "22M tall thin male brought in by EMS after sudden onset right chest pain and severe difficulty breathing while playing basketball. Rapidly progressive dyspnea over 20 minutes.",
    "ROS": {"Resp": "severe dyspnea, right chest pain", "CV": "lightheadedness", "Neuro": "anxiety, diaphoresis"},
    "Past_Medical_History": "No prior medical history. Tall thin build (6'4\", 155 lbs). No prior pneumothorax.",
    "Medications": "None",
    "Allergies": "NKDA",
    "Social_History": "College student, basketball player, non-smoker, social drinker",
    "Physical_Examination": "Severe respiratory distress, diaphoretic, cyanotic. Trachea deviated to left. Absent breath sounds right hemithorax. Hyperresonant to percussion right side. JVD present.",
}

DISEASES_DB["Severe Asthma Exacerbation"] = {
    "true_disease": "Severe Asthma Exacerbation",
    "true_symptoms": ["severe wheezing", "dyspnea", "inability to speak in full sentences", "accessory muscle use", "chest tightness"],
    "correct_treatment": "continuous nebulized albuterol, ipratropium, IV methylprednisolone 125mg, magnesium sulfate 2g IV, monitor for intubation",
    "lethal_treatments": ["sedatives without airway control", "beta-blockers"],
    "medical_history": "Asthma, prior intubation",
    "difficulty": "medium",
    "critical_labs": ["ABG", "peak_flow", "CXR"],
}
VITALS_DB["Severe Asthma Exacerbation"] = "HR 125, BP 140/85, RR 32, SpO2 88%, Temp 37.2C — tachycardic, tachypneic, hypoxic, using accessory muscles"
LAB_RESULTS_DB["Severe Asthma Exacerbation"] = {
    "ABG": "pH 7.32, pCO2 48, pO2 58, HCO3 24 — respiratory acidosis (ominous — tiring out)",
    "peak_flow": "Peak flow: 120 L/min (30% of predicted — severe obstruction)",
    "CXR": "Hyperinflated lungs, no pneumothorax, no infiltrate",
    "CBC": "WBC 14.0, Hgb 15.0, Plt 280 — stress response, hemoconcentration",
    "BMP": "Na 139, K 3.5, Cr 0.7 — hypokalemia from albuterol",
}
SOAP_HISTORY_DB["Severe Asthma Exacerbation"] = {
    "HPI": "28F with history of severe persistent asthma presents with worsening dyspnea and wheezing over 6 hours. Used albuterol inhaler 8 times with minimal relief. Cannot speak in full sentences. Had URI symptoms 3 days ago.",
    "ROS": {"Resp": "severe dyspnea, wheezing, chest tightness, unable to lie flat", "GI": "no complaints", "Neuro": "anxious, difficulty speaking"},
    "Past_Medical_History": "Severe persistent asthma since childhood, 2 prior ICU admissions, 1 intubation 3 years ago, allergic rhinitis",
    "Medications": "Fluticasone/salmeterol 500/50 BID, montelukast 10mg daily, albuterol PRN (using 8+ times today)",
    "Allergies": "Aspirin (bronchospasm), NSAIDs",
    "Social_History": "Daycare worker, non-smoker, has cat at home (known trigger), lives in older building with mold",
    "Physical_Examination": "Tripod position, using accessory muscles. Diffuse bilateral expiratory wheezing. Prolonged expiratory phase. Speaking in 2-3 word sentences. Diaphoretic.",
}

DISEASES_DB["COPD Exacerbation"] = {
    "true_disease": "COPD Exacerbation",
    "true_symptoms": ["worsening dyspnea", "increased sputum production", "purulent sputum", "cough", "wheezing"],
    "correct_treatment": "nebulized albuterol and ipratropium, prednisone 40mg oral, antibiotics azithromycin or doxycycline, supplemental oxygen titrated to SpO2 88-92%, BiPAP if needed",
    "lethal_treatments": ["high-flow oxygen without monitoring CO2 retention"],
    "medical_history": "COPD, smoking history",
    "difficulty": "easy",
    "critical_labs": ["ABG", "CXR", "CBC"],
}
VITALS_DB["COPD Exacerbation"] = "HR 100, BP 145/85, RR 26, SpO2 85% on RA, Temp 38.1C — hypoxic, low-grade fever"
LAB_RESULTS_DB["COPD Exacerbation"] = {
    "ABG": "pH 7.33, pCO2 58, pO2 55, HCO3 30 — chronic respiratory acidosis with acute worsening",
    "CXR": "Hyperinflated lungs, flattened diaphragms, no consolidation, no pneumothorax",
    "CBC": "WBC 13.5, Hgb 16.5 (polycythemia from chronic hypoxia), Plt 200",
    "BMP": "Na 140, K 4.2, Cr 1.1 — normal",
    "procalcitonin": "Procalcitonin: 0.3 ng/mL (low — viral or mild bacterial trigger)",
}
SOAP_HISTORY_DB["COPD Exacerbation"] = {
    "HPI": "68M with severe COPD presents with 3 days of worsening dyspnea, productive cough with green sputum, and increased wheeze. Uses home oxygen 2L NC baseline. Noticed increased sputum volume yesterday. Had similar episode 4 months ago.",
    "ROS": {"Resp": "worsening dyspnea, productive cough, green sputum, wheeze", "GI": "decreased appetite", "Neuro": "mild confusion this morning"},
    "Past_Medical_History": "Severe COPD GOLD stage III, 3 exacerbations in past year, home oxygen 2L, cor pulmonale",
    "Medications": "Tiotropium 18mcg inhaled daily, albuterol PRN, home O2 2L NC, prednisone taper recently completed",
    "Allergies": "Penicillin (rash)",
    "Social_History": "Retired truck driver, 50 pack-year smoking history quit 2 years ago, lives alone, limited mobility",
    "Physical_Examination": "Barrel chest, pursed-lip breathing, using accessory muscles. Diffuse rhonchi and wheezing bilaterally. Prolonged expiratory phase. Mild peripheral edema. Mildly confused.",
}

DISEASES_DB["Acute Respiratory Distress Syndrome"] = {
    "true_disease": "Acute Respiratory Distress Syndrome",
    "true_symptoms": ["severe progressive dyspnea", "refractory hypoxemia", "bilateral crackles", "tachypnea", "cyanosis"],
    "correct_treatment": "intubation and mechanical ventilation with low tidal volume 6mL/kg, PEEP optimization, prone positioning, treat underlying cause, restrictive fluid strategy",
    "lethal_treatments": ["high tidal volume ventilation", "aggressive fluid resuscitation"],
    "medical_history": "Pneumonia, sepsis, aspiration",
    "difficulty": "hard",
    "critical_labs": ["ABG", "CXR", "CBC", "BMP"],
}
VITALS_DB["Acute Respiratory Distress Syndrome"] = "HR 120, BP 95/55, RR 38, SpO2 78% on 15L NRB, Temp 39.2C — refractory hypoxemia despite max O2"
LAB_RESULTS_DB["Acute Respiratory Distress Syndrome"] = {
    "ABG": "pH 7.25, pCO2 52, pO2 55 on 100% FiO2, HCO3 20 — P/F ratio <100 (severe ARDS)",
    "CXR": "Bilateral diffuse infiltrates, white-out lungs, no cardiomegaly — consistent with ARDS",
    "CBC": "WBC 22.0 (critical high), Hgb 11.0, Plt 95 (low)",
    "BMP": "Na 135, K 5.1, Cr 1.8, Lactate 4.5 — AKI, lactic acidosis",
    "procalcitonin": "Procalcitonin: 12.5 ng/mL (high — sepsis likely trigger)",
    "blood_culture": "Blood cultures pending",
}
SOAP_HISTORY_DB["Acute Respiratory Distress Syndrome"] = {
    "HPI": "45F brought in by EMS with severe respiratory distress. Husband reports she had pneumonia treated with oral antibiotics 5 days ago, progressively worsening despite treatment. Now unable to breathe even at rest.",
    "ROS": {"Resp": "severe dyspnea, refractory to supplemental O2", "CV": "tachycardia", "Neuro": "confused, drowsy", "GI": "no complaints"},
    "Past_Medical_History": "Community-acquired pneumonia diagnosed 5 days ago, Type 2 DM, Obesity BMI 38",
    "Medications": "Amoxicillin 500mg TID (started 5 days ago), Metformin 500mg BID",
    "Allergies": "NKDA",
    "Social_History": "Elementary school teacher, non-smoker, no alcohol, lives with husband and 2 children",
    "Physical_Examination": "Severe respiratory distress, cyanotic, using all accessory muscles. Bilateral coarse crackles throughout all lung fields. Tachycardic. Confused, GCS 13.",
}

# ---------------------------------------------------------------------------
# CLASS 3: NEUROLOGICAL (5 diseases)
# ---------------------------------------------------------------------------

DISEASES_DB["Acute Ischemic Stroke"] = {"true_disease": "Acute Ischemic Stroke", "true_symptoms": ["sudden facial droop", "arm weakness", "slurred speech", "vision changes", "severe headache"], "correct_treatment": "IV alteplase within 4.5 hour window, CT head to rule out hemorrhage, admit to stroke unit, aspirin after 24 hours", "lethal_treatments": ["tPA if hemorrhagic stroke", "anticoagulation before CT"], "medical_history": "AFib, HTN, prior TIA", "difficulty": "medium", "critical_labs": ["CT_head", "CBC", "BMP", "ECG"]}
VITALS_DB["Acute Ischemic Stroke"] = "HR 88 irregular, BP 185/105, RR 18, SpO2 97%, Temp 37.0C — hypertensive, irregular rhythm"
LAB_RESULTS_DB["Acute Ischemic Stroke"] = {"CT_head": "No acute hemorrhage. Subtle early ischemic changes in left MCA territory", "CBC": "WBC 8.0, Hgb 14.0, Plt 210, INR 1.0", "BMP": "Na 140, K 4.0, Glucose 145, Cr 1.0", "ECG": "Atrial fibrillation rate 88, no ST changes"}
SOAP_HISTORY_DB["Acute Ischemic Stroke"] = {"HPI": "71M found by wife with right-sided weakness and slurred speech. Last seen normal 1.5 hours ago. Unable to raise right arm. Right facial droop.", "ROS": {"Neuro": "right hemiparesis, aphasia, facial droop", "CV": "irregular heartbeat known", "Resp": "no complaints"}, "Past_Medical_History": "Atrial fibrillation (not on anticoagulation), HTN, prior TIA 6 months ago", "Medications": "Aspirin 81mg daily, Amlodipine 5mg daily", "Allergies": "NKDA", "Social_History": "Retired professor, non-smoker, social drinker", "Physical_Examination": "Right facial droop, right arm drift, expressive aphasia. NIHSS score 14. Left gaze preference. Right homonymous hemianopia."}

DISEASES_DB["Subarachnoid Hemorrhage"] = {"true_disease": "Subarachnoid Hemorrhage", "true_symptoms": ["thunderclap headache worst of life", "neck stiffness", "photophobia", "nausea and vomiting", "brief loss of consciousness"], "correct_treatment": "emergent CT head, if negative then lumbar puncture, neurosurgery consult, nimodipine for vasospasm prevention, BP control, EVD if hydrocephalus", "lethal_treatments": ["lumbar puncture before CT if signs of herniation", "anticoagulation", "aspirin"], "medical_history": "Polycystic kidney disease, family history of aneurysm", "difficulty": "hard", "critical_labs": ["CT_head", "CSF", "CBC"]}
VITALS_DB["Subarachnoid Hemorrhage"] = "HR 65, BP 195/110, RR 16, SpO2 98%, Temp 37.5C — hypertensive, Cushing response"
LAB_RESULTS_DB["Subarachnoid Hemorrhage"] = {"CT_head": "Diffuse subarachnoid blood in basal cisterns, early hydrocephalus — SAH", "CSF": "CSF: xanthochromia present, RBC 50000, WBC 10, glucose 55, protein 80 — consistent with SAH", "CBC": "WBC 12.0, Hgb 13.5, Plt 200, INR 1.0", "BMP": "Na 132 (low — SIADH), K 3.8, Cr 0.9"}
SOAP_HISTORY_DB["Subarachnoid Hemorrhage"] = {"HPI": "42F presents with sudden onset thunderclap headache she describes as the worst headache of her life. Onset during exercise. Brief LOC witnessed by gym partner. Vomited x3.", "ROS": {"Neuro": "severe headache, photophobia, neck stiffness, brief LOC", "GI": "vomiting x3", "CV": "no chest pain"}, "Past_Medical_History": "Polycystic kidney disease, migraines (but states this is different from usual migraines), mother died of ruptured aneurysm", "Medications": "Sumatriptan PRN for migraines, oral contraceptive pill", "Allergies": "NKDA", "Social_History": "Fitness instructor, non-smoker, occasional wine", "Physical_Examination": "Photophobic, nuchal rigidity. Kernig sign positive. GCS 14 (confused). No focal motor deficits. Fundoscopy shows subhyaloid hemorrhages."}

DISEASES_DB["Status Epilepticus"] = {"true_disease": "Status Epilepticus", "true_symptoms": ["continuous seizure activity over 5 minutes", "altered consciousness", "tonic-clonic movements", "cyanosis", "incontinence"], "correct_treatment": "IV lorazepam 4mg, if persistent then IV fosphenytoin 20mg/kg or levetiracetam 60mg/kg, secure airway, glucose check, prepare for intubation if refractory", "lethal_treatments": ["phenytoin IV push rapid rate causing cardiac arrest"], "medical_history": "Epilepsy, medication non-compliance", "difficulty": "medium", "critical_labs": ["BMP", "glucose", "CT_head", "CBC"]}
VITALS_DB["Status Epilepticus"] = "HR 130, BP 170/100, RR 8 (irregular), SpO2 84%, Temp 38.5C — actively seizing, hypoxic"
LAB_RESULTS_DB["Status Epilepticus"] = {"BMP": "Na 128 (low), K 5.5 (high — rhabdomyolysis), Glucose 45 (LOW — hypoglycemia), Cr 1.5", "glucose": "Bedside glucose: 45 mg/dL — HYPOGLYCEMIA (possible seizure trigger)", "CT_head": "CT Head: no acute intracranial pathology, old left temporal encephalomalacia", "CBC": "WBC 15.0, Hgb 14.0, Plt 180", "CK": "CK: 8500 U/L (elevated — rhabdomyolysis from prolonged seizure)"}
SOAP_HISTORY_DB["Status Epilepticus"] = {"HPI": "35M brought in by EMS actively seizing. Bystanders report continuous seizure activity for approximately 12 minutes. EMS administered midazolam 10mg IM with brief cessation then recurrence.", "ROS": {"Neuro": "continuous seizure, postictal between episodes, incontinent of urine"}, "Past_Medical_History": "Epilepsy diagnosed age 12, on levetiracetam (ran out 4 days ago), 2 prior episodes of status epilepticus", "Medications": "Levetiracetam 1500mg BID (non-compliant, ran out)", "Allergies": "Carbamazepine (SJS risk — HLA-B*1502 positive)", "Social_History": "Warehouse worker, occasional marijuana, no alcohol", "Physical_Examination": "Actively seizing — generalized tonic-clonic. Cyanotic. Tongue laceration. Incontinent. GCS 3 during seizure."}

DISEASES_DB["Bacterial Meningitis"] = {"true_disease": "Bacterial Meningitis", "true_symptoms": ["severe headache", "high fever", "neck stiffness", "photophobia", "altered mental status"], "correct_treatment": "empiric IV ceftriaxone 2g plus vancomycin plus dexamethasone, lumbar puncture after CT if no contraindications, ICU admission", "lethal_treatments": ["delaying antibiotics for LP or imaging"], "medical_history": "Recent sinusitis, immunocompromised", "difficulty": "medium", "critical_labs": ["CSF", "CT_head", "CBC", "blood_culture"]}
VITALS_DB["Bacterial Meningitis"] = "HR 115, BP 90/55, RR 22, SpO2 96%, Temp 39.8C — febrile, tachycardic, borderline hypotensive"
LAB_RESULTS_DB["Bacterial Meningitis"] = {"CSF": "CSF: WBC 2500 (95% PMNs), Glucose 20 (LOW), Protein 450 (HIGH), Gram stain: Gram-positive diplococci — S. pneumoniae", "CT_head": "CT Head: mild meningeal enhancement, no mass effect, safe for LP", "CBC": "WBC 22.0 (left shift, 15% bands), Hgb 13.0, Plt 140", "blood_culture": "Blood cultures: Gram-positive diplococci growing at 8 hours", "BMP": "Na 130, K 4.5, Cr 1.3, Glucose 180"}
SOAP_HISTORY_DB["Bacterial Meningitis"] = {"HPI": "19M college student presents with 18 hours of severe headache, high fever, and neck stiffness. Roommate reports increasing confusion over the past 6 hours. Developed petechial rash on torso this morning.", "ROS": {"Neuro": "headache, confusion, photophobia, neck stiffness", "Derm": "petechial rash on torso", "GI": "vomiting x4"}, "Past_Medical_History": "Previously healthy, no immunodeficiency, did not receive meningococcal booster", "Medications": "None", "Allergies": "NKDA", "Social_History": "College freshman, lives in dormitory, roommate had URI last week", "Physical_Examination": "Toxic-appearing, confused. Nuchal rigidity. Kernig and Brudzinski signs positive. Petechial rash on trunk and extremities. GCS 12."}

DISEASES_DB["Guillain-Barre Syndrome"] = {"true_disease": "Guillain-Barre Syndrome", "true_symptoms": ["ascending bilateral weakness", "areflexia", "tingling in hands and feet", "back pain", "difficulty walking"], "correct_treatment": "IVIG 0.4g/kg/day for 5 days or plasmapheresis, monitor respiratory function with serial FVC, ICU admission if FVC declining, DVT prophylaxis", "lethal_treatments": ["corticosteroids as primary treatment"], "medical_history": "Recent viral illness or Campylobacter", "difficulty": "hard", "critical_labs": ["CSF", "CBC", "BMP", "peak_flow"]}
VITALS_DB["Guillain-Barre Syndrome"] = "HR 55 (bradycardia — dysautonomia), BP 160/95 labile, RR 22, SpO2 95%, Temp 37.0C — dysautonomia"
LAB_RESULTS_DB["Guillain-Barre Syndrome"] = {"CSF": "CSF: WBC 3 (normal), Protein 185 (HIGH), Glucose 65 (normal) — albuminocytologic dissociation, classic for GBS", "CBC": "WBC 7.5, Hgb 14.0, Plt 220 — normal", "BMP": "Na 140, K 4.0, Cr 0.9 — normal", "peak_flow": "FVC: 1.8L (45% predicted — declining, approaching intubation threshold)"}
SOAP_HISTORY_DB["Guillain-Barre Syndrome"] = {"HPI": "38M presents with 4 days of progressive bilateral leg weakness ascending to involve arms. Started as tingling in feet, now cannot walk unassisted. Had gastroenteritis with bloody diarrhea 2 weeks ago.", "ROS": {"Neuro": "ascending weakness, paresthesias, areflexia", "Resp": "mild dyspnea when lying flat, weak cough", "MSK": "low back pain", "GI": "resolved diarrhea 2 weeks ago"}, "Past_Medical_History": "Campylobacter gastroenteritis 2 weeks ago confirmed by stool culture, otherwise healthy", "Medications": "Completed azithromycin course for Campylobacter", "Allergies": "NKDA", "Social_History": "Software engineer, ate undercooked chicken at barbecue 3 weeks ago", "Physical_Examination": "Cannot stand unassisted. Bilateral symmetric weakness: legs 2/5, arms 3/5. Areflexia throughout. Sensation decreased in glove-stocking pattern. Facial weakness bilateral. Weak cough. FVC 1.8L."}

# ---------------------------------------------------------------------------
# CLASS 4: GASTROINTESTINAL (5 diseases)
# ---------------------------------------------------------------------------

DISEASES_DB["Upper GI Bleed"] = {"true_disease": "Upper GI Bleed", "true_symptoms": ["vomiting bright red blood", "melena", "lightheadedness", "epigastric pain", "progressive weakness"], "correct_treatment": "IV proton pump inhibitor pantoprazole bolus then drip, two large bore IVs with normal saline, type and crossmatch for transfusion, emergent GI consult for endoscopy", "lethal_treatments": ["NSAIDs", "anticoagulation without controlling bleed"], "medical_history": "NSAID use, alcohol abuse, peptic ulcer", "difficulty": "medium", "critical_labs": ["CBC", "BMP", "coagulation", "type_and_screen"]}
VITALS_DB["Upper GI Bleed"] = "HR 125, BP 80/50, RR 22, SpO2 96%, Temp 36.6C — tachycardic, hypotensive (hemorrhagic shock)"
LAB_RESULTS_DB["Upper GI Bleed"] = {"CBC": "WBC 10.2, Hgb 6.8 (CRITICAL LOW), Plt 160 — severe anemia from acute blood loss", "BMP": "Na 140, K 3.5, Cr 1.8, BUN 55 (elevated BUN:Cr ratio — upper GI source)", "coagulation": "PT 14, INR 1.2, aPTT 32 — mildly prolonged", "type_and_screen": "Type O positive, antibody screen negative, crossmatch 4 units pRBC", "lactate": "Lactate: 4.2 mmol/L (elevated — tissue hypoperfusion)"}
SOAP_HISTORY_DB["Upper GI Bleed"] = {"HPI": "55M presents with 4 episodes of vomiting large amounts of bright red blood over the past 6 hours. Reports black tarry stools for 2 days. Progressive weakness and lightheadedness. Epigastric pain worsening over 1 week.", "ROS": {"GI": "hematemesis, melena, epigastric pain", "CV": "lightheadedness, near-syncope", "Neuro": "weakness"}, "Past_Medical_History": "Peptic ulcer disease, chronic NSAID use for back pain, alcohol use disorder", "Medications": "Ibuprofen 800mg TID, no PPI prescribed", "Allergies": "NKDA", "Social_History": "Construction worker, drinks 6-8 beers daily, smokes 1 pack/day", "Physical_Examination": "Pale, diaphoretic, tachycardic. Abdomen tender in epigastrium. Rectal exam: melena confirmed. Orthostatic: HR increases 30bpm on standing."}

DISEASES_DB["Acute Appendicitis"] = {"true_disease": "Acute Appendicitis", "true_symptoms": ["periumbilical pain migrating to RLQ", "nausea", "anorexia", "low-grade fever", "rebound tenderness"], "correct_treatment": "IV antibiotics cefoxitin or piperacillin-tazobactam, emergent surgical consult for appendectomy, NPO, IV fluids, pain management", "lethal_treatments": ["delaying surgery if perforation suspected"], "medical_history": "None specific", "difficulty": "easy", "critical_labs": ["CBC", "CT_abdomen", "urinalysis"]}
VITALS_DB["Acute Appendicitis"] = "HR 95, BP 125/80, RR 18, SpO2 99%, Temp 38.4C — low-grade fever, mild tachycardia"
LAB_RESULTS_DB["Acute Appendicitis"] = {"CBC": "WBC 15.5 (left shift, 10% bands), Hgb 14.0, Plt 250", "CT_abdomen": "CT Abdomen: dilated appendix 12mm with periappendiceal fat stranding, appendicolith present — acute appendicitis", "urinalysis": "Urinalysis: WBC 2, RBC 5 — essentially normal (rules out UTI/stone)", "BMP": "Na 139, K 4.0, Cr 0.8 — normal", "lipase": "Lipase: 30 U/L — normal (rules out pancreatitis)"}
SOAP_HISTORY_DB["Acute Appendicitis"] = {"HPI": "24M presents with 18 hours of abdominal pain. Started periumbilically, now localized to right lower quadrant. Associated nausea and anorexia. Pain worsened with movement and coughing.", "ROS": {"GI": "RLQ pain, nausea, anorexia, no vomiting", "GU": "no dysuria", "Neuro": "no complaints"}, "Past_Medical_History": "Previously healthy, no prior surgeries", "Medications": "None", "Allergies": "NKDA", "Social_History": "College student, non-smoker, social drinker", "Physical_Examination": "Guarding in RLQ. McBurney point tenderness. Positive Rovsing sign. Positive psoas sign. Rebound tenderness present. Low-grade fever."}

DISEASES_DB["Acute Pancreatitis"] = {"true_disease": "Acute Pancreatitis", "true_symptoms": ["severe epigastric pain radiating to back", "nausea and vomiting", "abdominal distension", "worse after eating", "fever"], "correct_treatment": "aggressive IV fluid resuscitation with lactated Ringers, NPO initially, pain control with IV hydromorphone or fentanyl, monitor for complications, CT if no improvement in 48-72 hours", "lethal_treatments": ["early surgical intervention without indication"], "medical_history": "Gallstones, alcohol abuse", "difficulty": "medium", "critical_labs": ["lipase", "CBC", "BMP", "CT_abdomen"]}
VITALS_DB["Acute Pancreatitis"] = "HR 110, BP 100/60, RR 20, SpO2 96%, Temp 38.2C — tachycardic, mildly hypotensive from third-spacing"
LAB_RESULTS_DB["Acute Pancreatitis"] = {"lipase": "Lipase: 1850 U/L (CRITICAL HIGH — >3x upper limit, diagnostic for pancreatitis)", "CBC": "WBC 16.0, Hgb 15.5 (hemoconcentration), Plt 210", "BMP": "Na 136, K 3.8, Cr 1.4, Ca 7.8 (LOW — hypocalcemia, poor prognostic sign), Glucose 220", "CT_abdomen": "CT Abdomen: enlarged edematous pancreas with peripancreatic fluid collection, no necrosis — acute interstitial pancreatitis", "LFTs": "AST 180, ALT 210, Alk Phos 280, T.Bili 2.5 — gallstone etiology likely"}
SOAP_HISTORY_DB["Acute Pancreatitis"] = {"HPI": "48F presents with 12 hours of severe epigastric pain radiating straight through to her back. Rates pain 10/10. Vomited 5 times. Pain worse after eating dinner last night. Unable to find comfortable position.", "ROS": {"GI": "severe epigastric pain, vomiting, anorexia, abdominal distension", "CV": "no chest pain", "Resp": "mild dyspnea lying flat"}, "Past_Medical_History": "Gallstones diagnosed 6 months ago (declined cholecystectomy), obesity BMI 35, Type 2 DM", "Medications": "Metformin 1000mg BID, omeprazole 20mg daily", "Allergies": "Morphine (nausea)", "Social_History": "Office manager, non-smoker, 2 glasses wine on weekends", "Physical_Examination": "Writhing in pain, diaphoretic. Abdomen distended, tender in epigastrium with guarding. Decreased bowel sounds. No rebound. Mild jaundice."}

DISEASES_DB["Bowel Obstruction"] = {"true_disease": "Bowel Obstruction", "true_symptoms": ["colicky abdominal pain", "vomiting", "abdominal distension", "constipation and obstipation", "high-pitched bowel sounds"], "correct_treatment": "NPO, nasogastric tube decompression, IV fluid resuscitation, surgical consult, CT abdomen with contrast, monitor for strangulation signs", "lethal_treatments": ["barium enema if perforation suspected"], "medical_history": "Prior abdominal surgery, hernias", "difficulty": "medium", "critical_labs": ["CT_abdomen", "CBC", "BMP", "lactate"]}
VITALS_DB["Bowel Obstruction"] = "HR 105, BP 110/70, RR 20, SpO2 97%, Temp 37.8C — tachycardic from dehydration and pain"
LAB_RESULTS_DB["Bowel Obstruction"] = {"CT_abdomen": "CT Abdomen: dilated small bowel loops with transition point in right lower quadrant, consistent with adhesive small bowel obstruction, no free air", "CBC": "WBC 14.0, Hgb 16.0 (hemoconcentration), Plt 300", "BMP": "Na 132, K 3.2 (LOW), Cl 88 (LOW), Cr 1.5, BUN 40 — hypochloremic hypokalemic metabolic alkalosis from vomiting", "lactate": "Lactate: 2.5 mmol/L (mildly elevated — monitor for ischemia)"}
SOAP_HISTORY_DB["Bowel Obstruction"] = {"HPI": "65F presents with 2 days of progressive crampy abdominal pain, bilious vomiting, and inability to pass gas or stool. Abdomen progressively distending. Pain comes in waves every 5-10 minutes.", "ROS": {"GI": "crampy pain, vomiting, distension, obstipation", "GU": "decreased urine output"}, "Past_Medical_History": "Appendectomy 30 years ago, hysterectomy 15 years ago, prior episode of SBO managed conservatively", "Medications": "HCTZ 25mg daily, calcium supplement", "Allergies": "Codeine (vomiting)", "Social_History": "Retired nurse, non-smoker, no alcohol", "Physical_Examination": "Distended abdomen, diffusely tender. High-pitched tinkling bowel sounds. Midline surgical scar. No hernias palpable. Mild dehydration."}

DISEASES_DB["Cholecystitis"] = {"true_disease": "Cholecystitis", "true_symptoms": ["right upper quadrant pain", "pain after fatty meals", "nausea and vomiting", "fever", "positive Murphy sign"], "correct_treatment": "IV antibiotics piperacillin-tazobactam, surgical consult for cholecystectomy within 72 hours, NPO, IV fluids, pain control with ketorolac", "lethal_treatments": ["NSAID if renal failure present"], "medical_history": "Gallstones, obesity, female", "difficulty": "easy", "critical_labs": ["ultrasound", "CBC", "LFTs"]}
VITALS_DB["Cholecystitis"] = "HR 98, BP 135/85, RR 18, SpO2 98%, Temp 38.6C — low-grade fever, mild tachycardia"
LAB_RESULTS_DB["Cholecystitis"] = {"ultrasound": "RUQ US: gallbladder wall thickening 6mm, pericholecystic fluid, multiple gallstones, positive sonographic Murphy sign — acute cholecystitis", "CBC": "WBC 14.5 (left shift), Hgb 13.5, Plt 280", "LFTs": "AST 55, ALT 65, Alk Phos 180, T.Bili 1.8 — mildly elevated, possible CBD stone", "BMP": "Na 140, K 4.0, Cr 0.9 — normal", "lipase": "Lipase: 45 U/L — normal"}
SOAP_HISTORY_DB["Cholecystitis"] = {"HPI": "42F presents with 8 hours of progressively worsening right upper quadrant pain radiating to right shoulder. Pain started 2 hours after eating fried chicken. Associated nausea and 2 episodes of vomiting.", "ROS": {"GI": "RUQ pain, nausea, vomiting, anorexia", "Resp": "pain with deep breathing"}, "Past_Medical_History": "Known gallstones found incidentally 1 year ago, obesity BMI 32, 3 prior pregnancies", "Medications": "Oral contraceptive pill", "Allergies": "NKDA", "Social_History": "Stay-at-home mother, non-smoker, no alcohol", "Physical_Examination": "RUQ tenderness with guarding. Positive Murphy sign (inspiratory arrest with RUQ palpation). No rebound. Mild jaundice. Bowel sounds present."}


# ---------------------------------------------------------------------------
# CLASS 5: ENDOCRINE / METABOLIC (5 diseases)
# ---------------------------------------------------------------------------

DISEASES_DB["Diabetic Ketoacidosis"] = {"true_disease": "Diabetic Ketoacidosis", "true_symptoms": ["nausea and vomiting", "abdominal pain", "fruity breath", "Kussmaul breathing", "polyuria and polydipsia"], "correct_treatment": "IV insulin drip 0.1 units/kg/hr, aggressive IV normal saline, potassium replacement, monitor glucose hourly, search for precipitant", "lethal_treatments": ["IV insulin bolus without checking potassium first", "bicarbonate unless pH below 6.9"], "medical_history": "Type 1 DM, insulin non-compliance", "difficulty": "medium", "critical_labs": ["BMP", "ABG", "CBC", "urinalysis"]}
VITALS_DB["Diabetic Ketoacidosis"] = "HR 120, BP 95/55, RR 32 deep Kussmaul, SpO2 98%, Temp 37.8C  --  tachycardic, hypotensive, Kussmaul respirations"
LAB_RESULTS_DB["Diabetic Ketoacidosis"] = {"BMP": "Na 128, K 5.8 (HIGH but total body K depleted), Cl 95, CO2 8 (LOW), Glucose 520, Cr 1.8, Anion gap 25", "ABG": "pH 7.12, pCO2 18, pO2 98, HCO3 6  --  severe metabolic acidosis with anion gap", "CBC": "WBC 18.0 (stress response), Hgb 16.0 (hemoconcentration), Plt 250", "urinalysis": "Urinalysis: glucose 4+, ketones 4+, specific gravity 1.035  --  consistent with DKA"}
SOAP_HISTORY_DB["Diabetic Ketoacidosis"] = {"HPI": "22F with Type 1 DM presents with 2 days of nausea, vomiting, diffuse abdominal pain, and increasing confusion. Roommate reports she ran out of insulin 3 days ago. Fruity odor on breath noted.", "ROS": {"GI": "nausea, vomiting, abdominal pain", "Resp": "deep rapid breathing", "Neuro": "confusion, lethargy", "GU": "polyuria, polydipsia x 3 days"}, "Past_Medical_History": "Type 1 DM diagnosed age 14, prior DKA admission 2 years ago, depression", "Medications": "Insulin glargine 20u nightly, insulin lispro sliding scale (ran out 3 days ago), sertraline 50mg", "Allergies": "NKDA", "Social_History": "College student, non-smoker, social drinker, lives in dorm", "Physical_Examination": "Lethargic, dry mucous membranes, poor skin turgor. Fruity breath. Kussmaul respirations. Abdomen diffusely tender without peritoneal signs. Tachycardic."}

DISEASES_DB["Thyroid Storm"] = {"true_disease": "Thyroid Storm", "true_symptoms": ["high fever", "tachycardia out of proportion", "agitation and delirium", "tremor", "diarrhea"], "correct_treatment": "propylthiouracil or methimazole, propranolol for rate control, hydrocortisone 100mg IV, cooling measures, ICU admission", "lethal_treatments": ["radioactive iodine acutely", "iodine before thionamide"], "medical_history": "Graves disease, hyperthyroidism", "difficulty": "hard", "critical_labs": ["TSH", "BMP", "CBC", "ECG"]}
VITALS_DB["Thyroid Storm"] = "HR 165, BP 160/60 (wide pulse pressure), RR 28, SpO2 97%, Temp 40.2C  --  extreme tachycardia, hyperthermia"
LAB_RESULTS_DB["Thyroid Storm"] = {"TSH": "TSH: <0.01 mIU/L (undetectable), Free T4: 7.8 ng/dL (CRITICAL HIGH), Free T3: 22 pg/mL (CRITICAL HIGH)", "BMP": "Na 135, K 3.2 (LOW), Glucose 250, Ca 11.5 (HIGH), Cr 1.0", "CBC": "WBC 12.0, Hgb 12.5, Plt 180", "ECG": "Sinus tachycardia rate 165, no ST changes, possible atrial fibrillation"}
SOAP_HISTORY_DB["Thyroid Storm"] = {"HPI": "35F presents with 2 days of worsening agitation, tremor, palpitations, and diarrhea. Temperature 40.2C at home. Husband reports she has been increasingly confused and combative. Recently stopped her thyroid medication.", "ROS": {"Neuro": "agitation, tremor, confusion", "CV": "palpitations, chest discomfort", "GI": "diarrhea x 5 episodes", "Derm": "diaphoresis, warm flushed skin"}, "Past_Medical_History": "Graves disease diagnosed 3 years ago, stopped methimazole 2 weeks ago due to side effects", "Medications": "Methimazole 10mg TID (discontinued 2 weeks ago)", "Allergies": "NKDA", "Social_History": "Marketing executive, non-smoker, no alcohol", "Physical_Examination": "Agitated, diaphoretic, tremulous. Exophthalmos bilateral. Thyroid diffusely enlarged with bruit. Tachycardic, wide pulse pressure. Hyperreflexia. Fever 40.2C."}

DISEASES_DB["Adrenal Crisis"] = {"true_disease": "Adrenal Crisis", "true_symptoms": ["severe hypotension refractory to fluids", "abdominal pain", "weakness and fatigue", "confusion", "nausea and vomiting"], "correct_treatment": "IV hydrocortisone 100mg stat then 50mg every 8 hours, aggressive IV normal saline, dextrose if hypoglycemic, treat precipitating cause", "lethal_treatments": ["vasopressors without steroids"], "medical_history": "Chronic steroid use, Addison disease", "difficulty": "hard", "critical_labs": ["cortisol", "BMP", "CBC"]}
VITALS_DB["Adrenal Crisis"] = "HR 130, BP 65/40 (refractory to fluids), RR 24, SpO2 96%, Temp 38.5C  --  profound hypotension"
LAB_RESULTS_DB["Adrenal Crisis"] = {"cortisol": "Random cortisol: 1.2 mcg/dL (CRITICAL LOW  --  should be >18 in stress)", "BMP": "Na 118 (CRITICAL LOW), K 6.2 (HIGH), Glucose 45 (LOW), Cr 1.5", "CBC": "WBC 3.5 (LOW), Hgb 11.0, Plt 150, eosinophilia 12%", "ACTH": "ACTH: 450 pg/mL (elevated  --  primary adrenal insufficiency)"}
SOAP_HISTORY_DB["Adrenal Crisis"] = {"HPI": "52M presents with progressive weakness, nausea, and abdominal pain over 24 hours. Became confused and near-syncopal this morning. Has been on chronic prednisone which was abruptly stopped 5 days ago by another provider.", "ROS": {"CV": "lightheadedness, near-syncope", "GI": "nausea, vomiting, abdominal pain", "Neuro": "confusion, weakness", "Derm": "skin hyperpigmentation noted"}, "Past_Medical_History": "Rheumatoid arthritis on chronic prednisone 20mg daily x 3 years, abruptly discontinued 5 days ago", "Medications": "Prednisone 20mg daily (STOPPED 5 days ago), methotrexate 15mg weekly", "Allergies": "NKDA", "Social_History": "Retired, non-smoker, no alcohol", "Physical_Examination": "Obtunded, hyperpigmented skin creases and buccal mucosa. Profoundly hypotensive despite 2L NS. Abdomen tender diffusely. Weak pulses."}

DISEASES_DB["Severe Hypoglycemia"] = {"true_disease": "Severe Hypoglycemia", "true_symptoms": ["confusion", "diaphoresis", "tremor", "seizure", "loss of consciousness"], "correct_treatment": "IV dextrose D50 25g (50mL) stat, glucagon 1mg IM if no IV access, recheck glucose in 15 minutes, determine and treat cause", "lethal_treatments": ["insulin administration"], "medical_history": "Diabetes on insulin or sulfonylureas", "difficulty": "easy", "critical_labs": ["glucose", "BMP", "CBC"]}
VITALS_DB["Severe Hypoglycemia"] = "HR 110, BP 150/90, RR 20, SpO2 98%, Temp 36.5C  --  tachycardic, hypertensive (catecholamine surge)"
LAB_RESULTS_DB["Severe Hypoglycemia"] = {"glucose": "Bedside glucose: 28 mg/dL (CRITICAL LOW)", "BMP": "Na 140, K 4.0, Glucose 28 (CRITICAL LOW), Cr 1.8 (CKD  --  reduced insulin clearance)", "CBC": "WBC 8.0, Hgb 10.5, Plt 200  --  mild anemia of CKD"}
SOAP_HISTORY_DB["Severe Hypoglycemia"] = {"HPI": "75M found by daughter unresponsive at home. Diaphoretic and tremulous. Daughter reports he took his insulin this morning but did not eat breakfast. History of similar episodes.", "ROS": {"Neuro": "unresponsive, diaphoresis, tremor"}, "Past_Medical_History": "Type 2 DM on insulin, CKD stage 3 (reduced insulin clearance), prior hypoglycemic episodes", "Medications": "Insulin glargine 30u nightly, glipizide 10mg BID, metformin 500mg BID (should be held for CKD)", "Allergies": "NKDA", "Social_History": "Retired, lives alone, daughter visits daily, poor appetite recently", "Physical_Examination": "Unresponsive, GCS 6. Diaphoretic, cool clammy skin. Tremor. No focal neurological deficits. Pupils equal and reactive."}

DISEASES_DB["Hyperkalemia"] = {"true_disease": "Hyperkalemia", "true_symptoms": ["muscle weakness", "palpitations", "chest pain", "paresthesias", "nausea"], "correct_treatment": "IV calcium gluconate 10mL for cardiac stabilization, IV insulin 10 units with D50, albuterol nebulizer, kayexalate or patiromer, emergent dialysis if refractory", "lethal_treatments": ["calcium chloride via peripheral IV (extravasation necrosis)"], "medical_history": "CKD, ACE inhibitor use, potassium supplements", "difficulty": "medium", "critical_labs": ["BMP", "ECG", "CBC"]}
VITALS_DB["Hyperkalemia"] = "HR 45 (bradycardia), BP 100/65, RR 18, SpO2 97%, Temp 36.8C  --  bradycardic"
LAB_RESULTS_DB["Hyperkalemia"] = {"BMP": "Na 132, K 7.8 (CRITICAL HIGH), Cr 5.5 (ESRD), BUN 85, CO2 16 (metabolic acidosis)", "ECG": "Peaked T waves, widened QRS, loss of P waves  --  CRITICAL: approaching sine wave pattern", "CBC": "WBC 7.0, Hgb 9.0 (anemia of CKD), Plt 180"}
SOAP_HISTORY_DB["Hyperkalemia"] = {"HPI": "62M with ESRD on dialysis presents with 1 day of progressive weakness, palpitations, and nausea. Missed his last 2 dialysis sessions. Reports eating bananas and oranges heavily this week.", "ROS": {"CV": "palpitations, chest discomfort", "MSK": "generalized weakness", "GI": "nausea", "Neuro": "tingling in fingers"}, "Past_Medical_History": "ESRD on hemodialysis MWF, missed last 2 sessions, HTN, Type 2 DM", "Medications": "Lisinopril 40mg daily, sevelamer, EPO injections, potassium supplement (should have been stopped)", "Allergies": "NKDA", "Social_History": "Retired, lives with wife, transportation issues to dialysis center", "Physical_Examination": "Lethargic, bradycardic. Generalized muscle weakness 3/5 throughout. AV fistula left arm with good thrill. Mild peripheral edema."}

# ---------------------------------------------------------------------------
# CLASS 6: TOXICOLOGY (5 diseases)
# ---------------------------------------------------------------------------

DISEASES_DB["Opioid Overdose"] = {"true_disease": "Opioid Overdose", "true_symptoms": ["pinpoint pupils", "respiratory depression", "altered consciousness", "cyanosis", "bradycardia"], "correct_treatment": "naloxone 0.4mg IV repeat every 2-3 minutes, bag-valve mask ventilation, intubation if no response, monitor for re-sedation", "lethal_treatments": ["sedatives", "benzodiazepines"], "medical_history": "Opioid use disorder, chronic pain", "difficulty": "easy", "critical_labs": ["urine_tox", "ABG", "CBC"]}
VITALS_DB["Opioid Overdose"] = "HR 50, BP 85/50, RR 4, SpO2 72%, Temp 35.8C -- bradycardic, severe respiratory depression, hypothermic"
LAB_RESULTS_DB["Opioid Overdose"] = {"urine_tox": "Urine tox screen: positive for opioids, negative for benzos/amphetamines/cocaine", "ABG": "pH 7.18, pCO2 75, pO2 42, HCO3 24 -- respiratory acidosis from hypoventilation", "CBC": "WBC 7.0, Hgb 13.0, Plt 200 -- normal"}
SOAP_HISTORY_DB["Opioid Overdose"] = {"HPI": "28M found unresponsive by friends at home. Needle and drug paraphernalia nearby. Agonal respirations. Friends report heroin use.", "ROS": {"Neuro": "unresponsive", "Resp": "agonal breathing"}, "Past_Medical_History": "Opioid use disorder, prior overdose x2, hepatitis C", "Medications": "None prescribed", "Allergies": "NKDA", "Social_History": "Unemployed, IV heroin user x 5 years, lives in shelter", "Physical_Examination": "Unresponsive, GCS 3. Pinpoint pupils. RR 4, cyanotic. Track marks bilateral arms. No trauma."}

DISEASES_DB["Acetaminophen Toxicity"] = {"true_disease": "Acetaminophen Toxicity", "true_symptoms": ["nausea and vomiting", "right upper quadrant pain", "jaundice", "confusion", "malaise"], "correct_treatment": "N-acetylcysteine IV protocol 150mg/kg loading then 50mg/kg over 4h then 100mg/kg over 16h, acetaminophen level, LFTs serial, poison control consult", "lethal_treatments": ["delaying NAC beyond 8 hours post ingestion"], "medical_history": "Depression, intentional ingestion", "difficulty": "medium", "critical_labs": ["acetaminophen_level", "LFTs", "BMP", "coagulation"]}
VITALS_DB["Acetaminophen Toxicity"] = "HR 95, BP 110/70, RR 18, SpO2 99%, Temp 37.0C -- initially stable (deceptive)"
LAB_RESULTS_DB["Acetaminophen Toxicity"] = {"acetaminophen_level": "Acetaminophen level: 180 mcg/mL at 4 hours post ingestion (ABOVE Rumack-Matthew treatment line)", "LFTs": "AST 85, ALT 92, Alk Phos 120 -- early elevation, expect massive rise", "BMP": "Na 140, K 4.0, Cr 1.0, Glucose 95 -- normal early", "coagulation": "PT 14, INR 1.3 -- early coagulopathy developing"}
SOAP_HISTORY_DB["Acetaminophen Toxicity"] = {"HPI": "19F brought in by parents after admitting to ingesting approximately 50 tablets of extra-strength Tylenol (500mg each = ~25g) approximately 6 hours ago after argument with boyfriend. Currently nauseous with RUQ discomfort.", "ROS": {"GI": "nausea, vomiting, RUQ pain", "Psych": "suicidal ideation, regretful", "Neuro": "mild malaise"}, "Past_Medical_History": "Depression, anxiety, no prior suicide attempts", "Medications": "Sertraline 100mg daily", "Allergies": "NKDA", "Social_History": "College student, lives with parents, recently broken up with boyfriend", "Physical_Examination": "Tearful, cooperative. RUQ mildly tender. No jaundice yet. Alert and oriented. No focal deficits."}

DISEASES_DB["Carbon Monoxide Poisoning"] = {"true_disease": "Carbon Monoxide Poisoning", "true_symptoms": ["headache", "confusion", "cherry red skin", "nausea", "dizziness"], "correct_treatment": "100% oxygen via non-rebreather mask, consider hyperbaric oxygen if COHb above 25% or neurologic symptoms or pregnancy, serial COHb levels", "lethal_treatments": ["relying on pulse oximetry alone (falsely normal in CO poisoning)"], "medical_history": "Faulty heater, house fire, enclosed space", "difficulty": "medium", "critical_labs": ["COHb", "ABG", "ECG", "BMP"]}
VITALS_DB["Carbon Monoxide Poisoning"] = "HR 105, BP 130/80, RR 22, SpO2 98% (FALSELY NORMAL), Temp 37.0C -- SpO2 unreliable in CO poisoning!"
LAB_RESULTS_DB["Carbon Monoxide Poisoning"] = {"COHb": "Carboxyhemoglobin: 32% (CRITICAL -- severe CO poisoning, >25% requires hyperbaric)", "ABG": "pH 7.30, pCO2 32, pO2 85 (misleading), HCO3 18 -- metabolic acidosis, lactate 5.2", "ECG": "Sinus tachycardia, diffuse ST depression -- myocardial ischemia from CO", "BMP": "Na 140, K 4.5, Cr 1.0, Glucose 160"}
SOAP_HISTORY_DB["Carbon Monoxide Poisoning"] = {"HPI": "Family of 4 (father 45M presenting) brought in by fire department after found confused in home with headaches. Gas heater was running in closed room overnight. All family members symptomatic.", "ROS": {"Neuro": "headache, confusion, dizziness", "CV": "chest tightness", "GI": "nausea"}, "Past_Medical_History": "Healthy, no chronic conditions", "Medications": "None", "Allergies": "NKDA", "Social_History": "Factory worker, lives in older home with gas heating, wife and 2 children also symptomatic", "Physical_Examination": "Confused, cherry-red discoloration of lips. SpO2 reads 98% (unreliable). Tachycardic. Mild ataxia on gait testing."}

DISEASES_DB["Alcohol Withdrawal"] = {"true_disease": "Alcohol Withdrawal", "true_symptoms": ["tremor", "agitation", "hallucinations", "tachycardia", "diaphoresis"], "correct_treatment": "IV diazepam or lorazepam using CIWA protocol, thiamine 500mg IV before glucose, folate, magnesium replacement, monitor for seizures and delirium tremens", "lethal_treatments": ["IV glucose before thiamine (precipitates Wernicke)"], "medical_history": "Heavy alcohol use, prior withdrawal seizures", "difficulty": "medium", "critical_labs": ["BMP", "CBC", "LFTs", "ethanol_level"]}
VITALS_DB["Alcohol Withdrawal"] = "HR 125, BP 170/100, RR 22, SpO2 97%, Temp 38.3C -- tachycardic, hypertensive, low-grade fever"
LAB_RESULTS_DB["Alcohol Withdrawal"] = {"BMP": "Na 130, K 2.8 (LOW), Mg 1.0 (LOW), Glucose 65 (low), Cr 1.2", "CBC": "WBC 12.0, Hgb 10.5 (macrocytic), Plt 95 (low -- liver disease), MCV 108 (macrocytic)", "LFTs": "AST 220, ALT 95 (AST:ALT >2:1 -- alcoholic pattern), GGT 450, T.Bili 2.8", "ethanol_level": "Blood alcohol: 0 mg/dL (withdrawal occurring as alcohol cleared)"}
SOAP_HISTORY_DB["Alcohol Withdrawal"] = {"HPI": "52M presents with tremor, agitation, and visual hallucinations starting 48 hours after his last drink. Reports seeing spiders on walls. Last drink was 2 days ago when he ran out of money. History of heavy drinking 1 pint vodka daily x 20 years.", "ROS": {"Neuro": "tremor, agitation, visual hallucinations, insomnia", "CV": "palpitations", "GI": "nausea, anorexia"}, "Past_Medical_History": "Alcohol use disorder, alcoholic hepatitis, prior withdrawal seizure 1 year ago, malnutrition", "Medications": "None -- non-compliant with recommended medications", "Allergies": "NKDA", "Social_History": "Homeless, drinks 1 pint vodka daily x 20 years, smokes, no IV drug use", "Physical_Examination": "Agitated, tremulous, diaphoretic. Visual hallucinations (picking at sheets). Coarse hand tremor. Hepatomegaly. Spider angiomata. CIWA score 28 (severe)."}

DISEASES_DB["Serotonin Syndrome"] = {"true_disease": "Serotonin Syndrome", "true_symptoms": ["agitation", "hyperthermia", "clonus", "muscle rigidity", "diaphoresis"], "correct_treatment": "discontinue all serotonergic agents, cyproheptadine 12mg initial then 4mg every 2 hours, active cooling, benzodiazepines for agitation, ICU admission", "lethal_treatments": ["dantrolene (wrong diagnosis -- not NMS)", "additional serotonergic agents"], "medical_history": "Multiple serotonergic medications, recent dose change", "difficulty": "hard", "critical_labs": ["BMP", "CK", "CBC"]}
VITALS_DB["Serotonin Syndrome"] = "HR 135, BP 165/95, RR 26, SpO2 95%, Temp 39.8C -- hyperthermic, tachycardic, hypertensive"
LAB_RESULTS_DB["Serotonin Syndrome"] = {"BMP": "Na 138, K 4.8, Cr 1.5, Glucose 145", "CK": "CK: 3200 U/L (elevated -- muscle rigidity causing rhabdomyolysis)", "CBC": "WBC 14.0, Hgb 15.0 (hemoconcentration), Plt 200"}
SOAP_HISTORY_DB["Serotonin Syndrome"] = {"HPI": "34M presents with acute onset agitation, muscle rigidity, and fever starting 6 hours after his psychiatrist added tramadol to his existing SSRI regimen. Wife reports he became increasingly confused and developed jerking movements in his legs.", "ROS": {"Neuro": "agitation, confusion, jerking limb movements, muscle rigidity", "Derm": "profuse sweating", "GI": "diarrhea x3 episodes"}, "Past_Medical_History": "Major depressive disorder, chronic back pain, started tramadol today", "Medications": "Sertraline 200mg daily, trazodone 100mg nightly, tramadol 50mg TID (STARTED TODAY)", "Allergies": "NKDA", "Social_History": "Accountant, non-smoker, no alcohol", "Physical_Examination": "Agitated, diaphoretic, hyperthermic 39.8C. Bilateral lower extremity clonus (>10 beats). Muscle rigidity in legs. Hyperreflexia throughout. Dilated pupils. Tremor."}

# ---------------------------------------------------------------------------
# CLASS 7: TRAUMA (5 diseases)
# ---------------------------------------------------------------------------

DISEASES_DB["Traumatic Brain Injury"] = {"true_disease": "Traumatic Brain Injury", "true_symptoms": ["loss of consciousness", "confusion", "vomiting", "unequal pupils", "worsening headache"], "correct_treatment": "CT head emergent, neurosurgery consult, elevate head of bed 30 degrees, mannitol 1g/kg or hypertonic saline if herniating, intubation if GCS 8 or below", "lethal_treatments": ["lumbar puncture with elevated ICP", "anticoagulation acutely"], "medical_history": "Fall, assault, MVA", "difficulty": "medium", "critical_labs": ["CT_head", "CBC", "BMP", "coagulation"]}
VITALS_DB["Traumatic Brain Injury"] = "HR 55 (Cushing), BP 195/100 (Cushing), RR 10 irregular, SpO2 94%, Temp 37.0C -- Cushing triad concerning for herniation"
LAB_RESULTS_DB["Traumatic Brain Injury"] = {"CT_head": "CT Head: large right-sided epidural hematoma with 8mm midline shift, uncal herniation -- EMERGENT SURGICAL EVACUATION NEEDED", "CBC": "WBC 12.0, Hgb 12.5, Plt 220", "BMP": "Na 140, K 4.0, Cr 0.9 -- normal", "coagulation": "PT 12, INR 1.0, aPTT 28 -- normal"}
SOAP_HISTORY_DB["Traumatic Brain Injury"] = {"HPI": "35M brought by EMS after falling from 10-foot ladder at construction site. Witnessed brief LOC followed by lucid interval, now becoming progressively more confused and combative. Vomited x2 in ambulance.", "ROS": {"Neuro": "LOC, confusion, vomiting, combative"}, "Past_Medical_History": "Healthy, no bleeding disorders, no anticoagulant use", "Medications": "None", "Allergies": "NKDA", "Social_History": "Construction worker, non-smoker, social drinker, no helmet worn", "Physical_Examination": "GCS 9 (E2V3M4). Right temporal scalp hematoma. Right pupil 6mm fixed, left 3mm reactive. Left hemiparesis. Cushing triad present."}

DISEASES_DB["Open Femur Fracture"] = {"true_disease": "Open Femur Fracture", "true_symptoms": ["severe thigh pain", "visible bone through skin", "limb deformity", "significant bleeding", "inability to bear weight"], "correct_treatment": "tourniquet if active hemorrhage, IV fluid resuscitation, tetanus prophylaxis, IV cefazolin, emergent orthopedic consult, traction splint, pain management with IV fentanyl", "lethal_treatments": ["reducing open fracture in ED without OR"], "medical_history": "Trauma, MVA", "difficulty": "medium", "critical_labs": ["CBC", "BMP", "type_and_screen", "coagulation"]}
VITALS_DB["Open Femur Fracture"] = "HR 130, BP 80/50, RR 24, SpO2 97%, Temp 36.5C -- tachycardic, hypotensive from blood loss (up to 1500mL from femur)"
LAB_RESULTS_DB["Open Femur Fracture"] = {"CBC": "WBC 14.0, Hgb 8.5 (acute blood loss), Plt 200", "BMP": "Na 138, K 4.5, Cr 1.2, Lactate 3.8 -- lactic acidosis from hemorrhage", "type_and_screen": "Type A positive, crossmatch 4 units pRBC", "coagulation": "PT 13, INR 1.1, aPTT 30 -- normal"}
SOAP_HISTORY_DB["Open Femur Fracture"] = {"HPI": "25M motorcycle accident at high speed. Right thigh deformity with bone protruding through skin. Significant blood at scene. Screaming in pain. No LOC, no head injury.", "ROS": {"MSK": "right thigh pain, deformity, open wound", "CV": "lightheadedness"}, "Past_Medical_History": "Previously healthy", "Medications": "None", "Allergies": "NKDA", "Social_History": "College student, motorcycle rider, no helmet", "Physical_Examination": "Pale, diaphoretic, tachycardic. Right thigh: open fracture Gustilo type IIIA, bone visible, active bleeding. Right leg shortened and externally rotated. Distal pulses faint but present. Left leg normal. FAST scan negative."}

DISEASES_DB["Severe Burn Injury"] = {"true_disease": "Severe Burn Injury", "true_symptoms": ["burns over large body surface area", "pain or painless areas", "singed nasal hair", "hoarse voice", "soot in airway"], "correct_treatment": "secure airway early if inhalation injury suspected, Parkland formula IV fluids 4mL/kg per percent TBSA, wound care, tetanus, pain management, transfer to burn center", "lethal_treatments": ["delayed intubation with progressive airway edema"], "medical_history": "House fire, chemical exposure", "difficulty": "hard", "critical_labs": ["CBC", "BMP", "ABG", "COHb"]}
VITALS_DB["Severe Burn Injury"] = "HR 135, BP 90/55, RR 28, SpO2 93%, Temp 35.5C -- tachycardic, hypotensive from massive fluid loss, hypothermic"
LAB_RESULTS_DB["Severe Burn Injury"] = {"CBC": "WBC 18.0, Hgb 18.0 (hemoconcentration from plasma loss), Plt 300", "BMP": "Na 145 (high -- free water loss), K 5.5 (HIGH -- cell destruction), Cr 1.5, Glucose 200", "ABG": "pH 7.30, pCO2 35, pO2 70, HCO3 18, Lactate 5.0 -- metabolic acidosis", "COHb": "Carboxyhemoglobin: 15% (moderate -- inhalation injury likely)"}
SOAP_HISTORY_DB["Severe Burn Injury"] = {"HPI": "40M rescued from house fire by firefighters. Found in smoke-filled room. Burns to face, chest, bilateral arms. Hoarse voice and coughing soot. Burns estimated 35% TBSA mix of 2nd and 3rd degree.", "ROS": {"Resp": "hoarse voice, cough, soot in sputum", "Derm": "extensive burns face/chest/arms", "Neuro": "alert, severe pain in some areas, painless in others"}, "Past_Medical_History": "Healthy, no chronic conditions", "Medications": "None", "Allergies": "NKDA", "Social_History": "Electrician, smoker, fell asleep with cigarette at home", "Physical_Examination": "Burns: 2nd degree to face, anterior chest, bilateral arms. 3rd degree patches on chest (painless, waxy white). Singed nasal hairs, soot in oropharynx, stridor developing. TBSA approximately 35%."}

DISEASES_DB["Pelvic Fracture"] = {"true_disease": "Pelvic Fracture", "true_symptoms": ["pelvic pain", "inability to walk", "hemodynamic instability", "blood at urethral meatus", "lower abdominal pain"], "correct_treatment": "pelvic binder application, massive transfusion protocol, IR angiography for embolization if hemodynamically unstable, avoid Foley if blood at meatus, trauma surgery consult", "lethal_treatments": ["pelvic exam with rocking (worsens hemorrhage)", "Foley catheter if urethral injury suspected"], "medical_history": "High energy trauma, MVA, fall", "difficulty": "hard", "critical_labs": ["CBC", "type_and_screen", "CT_pelvis", "FAST"]}
VITALS_DB["Pelvic Fracture"] = "HR 140, BP 70/40, RR 28, SpO2 95%, Temp 35.8C -- hemorrhagic shock, hypothermic"
LAB_RESULTS_DB["Pelvic Fracture"] = {"CBC": "WBC 16.0, Hgb 7.0 (CRITICAL -- massive blood loss), Plt 110", "type_and_screen": "Type O negative, activate massive transfusion protocol", "CT_pelvis": "CT Pelvis: open-book pelvic fracture bilateral sacroiliac disruption, active arterial extravasation right internal iliac", "FAST": "FAST: positive for free fluid in pelvis, negative in Morrison pouch and splenorenal"}
SOAP_HISTORY_DB["Pelvic Fracture"] = {"HPI": "55F pedestrian struck by car at 40mph. Thrown 15 feet. Severe pelvic and lower abdominal pain. Unable to move legs. Blood noted at urethral meatus.", "ROS": {"MSK": "severe pelvic pain", "GU": "blood at meatus, unable to void", "CV": "lightheaded, thirsty"}, "Past_Medical_History": "Osteoporosis, on warfarin for DVT", "Medications": "Warfarin 5mg daily, calcium/vitamin D", "Allergies": "Codeine", "Social_History": "Retired teacher, was crossing street when struck", "Physical_Examination": "Pale, cold, diaphoretic. Pelvis unstable on gentle compression (do NOT repeat). Blood at urethral meatus. Ecchymosis perineum. Bilateral lower extremity sensation intact. Distal pulses weak."}

DISEASES_DB["Splenic Rupture"] = {"true_disease": "Splenic Rupture", "true_symptoms": ["left upper quadrant pain", "left shoulder pain Kehr sign", "abdominal rigidity", "hemodynamic instability", "history of abdominal trauma"], "correct_treatment": "emergent surgical consult, massive transfusion protocol if unstable, CT abdomen if stable enough, IR embolization for grade 3, splenectomy for grade 4-5 or unstable", "lethal_treatments": ["observation only if hemodynamically unstable"], "medical_history": "Blunt abdominal trauma, mononucleosis", "difficulty": "medium", "critical_labs": ["FAST", "CBC", "type_and_screen", "CT_abdomen"]}
VITALS_DB["Splenic Rupture"] = "HR 125, BP 85/50, RR 24, SpO2 96%, Temp 36.8C -- tachycardic, hypotensive from intra-abdominal hemorrhage"
LAB_RESULTS_DB["Splenic Rupture"] = {"FAST": "FAST: large amount of free fluid in left upper quadrant (splenorenal recess) and pelvis -- positive", "CBC": "WBC 15.0, Hgb 8.0 (dropping -- active hemorrhage), Plt 180", "type_and_screen": "Type B positive, crossmatch 6 units pRBC, activate MTP", "CT_abdomen": "CT Abdomen: Grade IV splenic laceration with active contrast extravasation, large hemoperitoneum"}
SOAP_HISTORY_DB["Splenic Rupture"] = {"HPI": "20M brought in after being tackled hard during football game 2 hours ago. Developed progressive LUQ abdominal pain radiating to left shoulder. Became lightheaded on the sideline then nearly passed out.", "ROS": {"GI": "LUQ pain, left shoulder pain", "CV": "lightheaded, near syncope"}, "Past_Medical_History": "Mononucleosis 3 weeks ago (splenomegaly noted on prior visit), cleared for sports by outside provider", "Medications": "None", "Allergies": "NKDA", "Social_History": "College football player, non-smoker, social drinker", "Physical_Examination": "Pale, diaphoretic, guarding LUQ. Kehr sign positive (left shoulder pain with LUQ palpation). Abdomen rigid LUQ. Rebound tenderness. Orthostatic hypotension."}

# ---------------------------------------------------------------------------
# CLASS 8: INFECTIOUS (5 diseases)
# ---------------------------------------------------------------------------

DISEASES_DB["Septic Shock"] = {"true_disease": "Septic Shock", "true_symptoms": ["high fever", "hypotension refractory to fluids", "tachycardia", "altered mental status", "warm flushed skin early then cold"], "correct_treatment": "IV broad spectrum antibiotics within 1 hour, 30mL/kg IV crystalloid bolus, norepinephrine if MAP below 65 after fluids, lactate monitoring, blood cultures before antibiotics, source control", "lethal_treatments": ["delaying antibiotics for cultures", "dopamine as first-line vasopressor"], "medical_history": "UTI, pneumonia, immunocompromised", "difficulty": "medium", "critical_labs": ["blood_culture", "lactate", "CBC", "BMP"]}
VITALS_DB["Septic Shock"] = "HR 130, BP 72/38 (MAP 49), RR 28, SpO2 93%, Temp 39.5C -- septic shock, MAP below 65"
LAB_RESULTS_DB["Septic Shock"] = {"blood_culture": "Blood cultures: Gram-negative rods growing at 6 hours -- E. coli", "lactate": "Lactate: 6.8 mmol/L (CRITICAL -- severe tissue hypoperfusion)", "CBC": "WBC 28.0 (critical, bandemia 20%), Hgb 11.0, Plt 65 (LOW -- DIC developing)", "BMP": "Na 132, K 5.0, Cr 2.5 (AKI), Glucose 180, CO2 14 (acidosis)"}
SOAP_HISTORY_DB["Septic Shock"] = {"HPI": "72F nursing home resident brought in with fever, confusion, and low blood pressure. Staff reports foul-smelling urine and decreased oral intake x 3 days.", "ROS": {"GU": "foul-smelling urine, frequency", "Neuro": "confusion, lethargy", "CV": "hypotension"}, "Past_Medical_History": "Type 2 DM, recurrent UTIs, Foley catheter, dementia", "Medications": "Metformin 500mg BID, donepezil 10mg daily", "Allergies": "Sulfa drugs", "Social_History": "Nursing home resident, non-ambulatory, Foley catheter", "Physical_Examination": "Obtunded, warm and flushed. Tachycardic. Hypotensive despite 1L NS. Suprapubic tenderness. Foley with cloudy malodorous urine. Mottled extremities."}

DISEASES_DB["Necrotizing Fasciitis"] = {"true_disease": "Necrotizing Fasciitis", "true_symptoms": ["pain out of proportion to exam", "rapidly spreading erythema", "crepitus", "bullae", "systemic toxicity"], "correct_treatment": "emergent surgical debridement, IV vancomycin plus piperacillin-tazobactam plus clindamycin, aggressive fluid resuscitation, ICU admission", "lethal_treatments": ["antibiotics alone without surgery", "observation"], "medical_history": "Diabetes, IV drug use, recent surgery", "difficulty": "hard", "critical_labs": ["CBC", "BMP", "CK", "lactate"]}
VITALS_DB["Necrotizing Fasciitis"] = "HR 135, BP 80/45, RR 26, SpO2 94%, Temp 39.8C -- septic, tachycardic"
LAB_RESULTS_DB["Necrotizing Fasciitis"] = {"CBC": "WBC 32.0 (CRITICAL), Hgb 12.0, Plt 80 (DIC)", "BMP": "Na 128, K 5.2, Cr 2.8 (AKI), Glucose 380", "CK": "CK: 12000 U/L (muscle destruction)", "lactate": "Lactate: 8.5 mmol/L (severe)"}
SOAP_HISTORY_DB["Necrotizing Fasciitis"] = {"HPI": "58M diabetic presents with 36 hours of rapidly worsening right lower leg pain, redness, and swelling. Pain is severe and out of proportion to visible findings. Small cut on shin 4 days ago. Developed dark blisters this morning.", "ROS": {"Derm": "severe leg pain, spreading redness, blisters", "Neuro": "confusion", "GI": "nausea"}, "Past_Medical_History": "Uncontrolled Type 2 DM A1c 11.2%, peripheral vascular disease, obesity", "Medications": "Metformin 1000mg BID, glipizide 10mg BID", "Allergies": "NKDA", "Social_History": "Retired, sedentary, non-smoker", "Physical_Examination": "Toxic-appearing. Right lower leg: tense edema, erythema extending rapidly (marked border advancing), hemorrhagic bullae, crepitus on palpation, pain out of proportion. Skin dusky/necrotic centrally."}

DISEASES_DB["Malaria"] = {"true_disease": "Malaria", "true_symptoms": ["cyclical high fevers", "rigors", "headache", "splenomegaly", "jaundice"], "correct_treatment": "IV artesunate for severe malaria, if uncomplicated then artemether-lumefantrine oral, monitor parasitemia every 12 hours, exchange transfusion if parasitemia above 10%", "lethal_treatments": ["chloroquine alone if P. falciparum resistant area"], "medical_history": "Travel to endemic area, no prophylaxis", "difficulty": "hard", "critical_labs": ["blood_smear", "CBC", "BMP", "LFTs"]}
VITALS_DB["Malaria"] = "HR 115, BP 95/60, RR 24, SpO2 95%, Temp 40.5C -- high fever with rigors"
LAB_RESULTS_DB["Malaria"] = {"blood_smear": "Thick and thin smear: Plasmodium falciparum, parasitemia 8%, ring forms and banana-shaped gametocytes", "CBC": "WBC 3.5 (LOW), Hgb 8.0 (severe anemia from hemolysis), Plt 35 (CRITICAL LOW)", "BMP": "Na 130, K 4.8, Cr 2.2 (AKI), Glucose 55 (LOW), T.Bili 5.5 (hemolysis)", "LFTs": "AST 180, ALT 120, LDH 850 (hemolysis)"}
SOAP_HISTORY_DB["Malaria"] = {"HPI": "30M presents with 5 days of cyclical high fevers with rigors every 48 hours, drenching sweats, headache, and progressive weakness. Returned from 3-week trip to sub-Saharan Africa 10 days ago. Did not take malaria prophylaxis.", "ROS": {"Neuro": "headache, confusion", "GI": "nausea, abdominal pain", "Derm": "jaundice"}, "Past_Medical_History": "Previously healthy, no prior malaria", "Medications": "Did not take prophylaxis -- was not prescribed", "Allergies": "NKDA", "Social_History": "NGO worker, traveled to rural Kenya/Tanzania, slept without bed nets", "Physical_Examination": "Jaundiced, febrile 40.5C with rigors. Splenomegaly 4cm below costal margin. Hepatomegaly. Pallor. Mildly confused. Petechiae on lower extremities."}

DISEASES_DB["Peritonsillar Abscess"] = {"true_disease": "Peritonsillar Abscess", "true_symptoms": ["severe sore throat unilateral", "trismus", "muffled hot potato voice", "drooling", "uvula deviation"], "correct_treatment": "needle aspiration or incision and drainage, IV clindamycin or ampicillin-sulbactam, dexamethasone, pain control, ENT consult", "lethal_treatments": ["blind intubation if airway compromise (risk of rupture)"], "medical_history": "Recent tonsillitis, incomplete antibiotic course", "difficulty": "easy", "critical_labs": ["CBC", "CT_neck"]}
VITALS_DB["Peritonsillar Abscess"] = "HR 100, BP 130/80, RR 18, SpO2 98%, Temp 38.8C -- febrile, mild tachycardia"
LAB_RESULTS_DB["Peritonsillar Abscess"] = {"CBC": "WBC 17.0 (left shift), Hgb 14.0, Plt 250", "CT_neck": "CT Neck with contrast: 3cm left peritonsillar abscess with rim enhancement, no extension to parapharyngeal space"}
SOAP_HISTORY_DB["Peritonsillar Abscess"] = {"HPI": "22M presents with 5 days of worsening left-sided sore throat, now unable to swallow. Progressive trismus -- cannot open mouth fully. Muffled voice. Drooling. Was treated for strep throat 1 week ago with 3 days of amoxicillin (did not finish course).", "ROS": {"ENT": "severe left throat pain, trismus, drooling, muffled voice", "Neuro": "no neck stiffness"}, "Past_Medical_History": "Recurrent tonsillitis x 3 episodes this year, incomplete antibiotic courses", "Medications": "Amoxicillin (stopped after 3 days of 10-day course)", "Allergies": "NKDA", "Social_History": "College student, smoker, social drinker", "Physical_Examination": "Drooling, muffled voice. Trismus (limited mouth opening). Left peritonsillar bulge with uvula deviated to right. Left tonsil displaced medially. No stridor. Neck supple, tender left submandibular lymphadenopathy."}

DISEASES_DB["Spontaneous Bacterial Peritonitis"] = {"true_disease": "Spontaneous Bacterial Peritonitis", "true_symptoms": ["abdominal pain and tenderness", "fever", "worsening ascites", "altered mental status", "diarrhea"], "correct_treatment": "IV cefotaxime 2g every 8 hours, IV albumin 1.5g/kg on day 1 and 1g/kg on day 3, diagnostic paracentesis, hepatology consult", "lethal_treatments": ["aminoglycosides in cirrhosis (nephrotoxicity)"], "medical_history": "Cirrhosis with ascites", "difficulty": "medium", "critical_labs": ["paracentesis", "CBC", "BMP", "blood_culture"]}
VITALS_DB["Spontaneous Bacterial Peritonitis"] = "HR 105, BP 90/55, RR 20, SpO2 96%, Temp 38.5C -- febrile, hypotensive"
LAB_RESULTS_DB["Spontaneous Bacterial Peritonitis"] = {"paracentesis": "Ascitic fluid: WBC 850 (PMN 680 -- above 250 threshold), protein 1.2, glucose 40, culture pending -- diagnostic of SBP", "CBC": "WBC 14.0, Hgb 9.0, Plt 55 (thrombocytopenia from liver disease)", "BMP": "Na 125 (dilutional), K 3.5, Cr 2.0 (hepatorenal), BUN 45", "blood_culture": "Blood cultures pending"}
SOAP_HISTORY_DB["Spontaneous Bacterial Peritonitis"] = {"HPI": "60M with decompensated cirrhosis presents with 2 days of worsening abdominal pain, distension, and fever. Reports increasing confusion per family. Ascites has been worsening despite diuretics.", "ROS": {"GI": "abdominal pain, distension, diarrhea", "Neuro": "confusion, worsening encephalopathy", "CV": "lightheadedness"}, "Past_Medical_History": "Alcoholic cirrhosis Child-Pugh C, recurrent ascites, prior SBP episode 6 months ago, esophageal varices", "Medications": "Spironolactone 100mg, furosemide 40mg, lactulose, rifaximin, nadolol", "Allergies": "NKDA", "Social_History": "Former heavy drinker (quit 1 year ago), retired, lives with adult daughter", "Physical_Examination": "Jaundiced, cachectic. Distended abdomen with tense ascites, diffusely tender with mild rebound. Shifting dullness positive. Spider angiomata. Asterixis present. Mild confusion."}

# ---------------------------------------------------------------------------
# CLASS 9: GENITOURINARY / RENAL (5 diseases)
# ---------------------------------------------------------------------------

DISEASES_DB["Acute Kidney Injury"] = {"true_disease": "Acute Kidney Injury", "true_symptoms": ["decreased urine output", "swelling", "nausea", "confusion", "shortness of breath"], "correct_treatment": "IV fluid resuscitation if prerenal, hold nephrotoxins, correct electrolytes, emergent dialysis if refractory hyperkalemia or pulmonary edema or uremia, nephrology consult", "lethal_treatments": ["NSAIDs", "IV contrast without indication", "potassium-containing fluids"], "medical_history": "Dehydration, sepsis, nephrotoxic medications", "difficulty": "medium", "critical_labs": ["BMP", "urinalysis", "CBC", "renal_US"]}
VITALS_DB["Acute Kidney Injury"] = "HR 95, BP 90/55, RR 22, SpO2 94%, Temp 37.5C -- hypotensive, mildly hypoxic from fluid overload"
LAB_RESULTS_DB["Acute Kidney Injury"] = {"BMP": "Na 130, K 6.5 (HIGH), Cr 5.8 (baseline 1.0 -- CRITICAL rise), BUN 80, CO2 14 (acidosis)", "urinalysis": "Urinalysis: muddy brown granular casts -- ATN (acute tubular necrosis)", "CBC": "WBC 12.0, Hgb 10.0, Plt 180", "renal_US": "Renal US: normal-sized kidneys, no hydronephrosis, no obstruction"}
SOAP_HISTORY_DB["Acute Kidney Injury"] = {"HPI": "68M presents with 2 days of minimal urine output, progressive swelling, and shortness of breath. Was treated with IV vancomycin and gentamicin for pneumonia last week at another hospital. Now confused.", "ROS": {"Renal": "oliguria, edema", "Resp": "dyspnea, cannot lie flat", "Neuro": "confusion", "GI": "nausea"}, "Past_Medical_History": "HTN, Type 2 DM, recent pneumonia treated with nephrotoxic antibiotics", "Medications": "Vancomycin (recent course), gentamicin (recent course), lisinopril 20mg, metformin", "Allergies": "NKDA", "Social_History": "Retired, lives with wife, non-smoker", "Physical_Examination": "Confused, edematous. JVD present. Bibasilar crackles. Abdomen mildly distended. 3+ pitting edema bilateral legs. Foley placed -- 50mL dark urine over 4 hours."}

DISEASES_DB["Nephrolithiasis"] = {"true_disease": "Nephrolithiasis", "true_symptoms": ["severe colicky flank pain", "hematuria", "nausea and vomiting", "pain radiating to groin", "restlessness"], "correct_treatment": "IV ketorolac 30mg for pain, IV ondansetron for nausea, IV fluids, CT abdomen without contrast, urology consult if stone greater than 6mm or signs of infection", "lethal_treatments": ["observation if obstructing stone with infection (sepsis risk)"], "medical_history": "Prior kidney stones, dehydration", "difficulty": "easy", "critical_labs": ["CT_abdomen", "urinalysis", "BMP"]}
VITALS_DB["Nephrolithiasis"] = "HR 100, BP 160/95 (pain), RR 20, SpO2 99%, Temp 37.0C -- tachycardic and hypertensive from pain"
LAB_RESULTS_DB["Nephrolithiasis"] = {"CT_abdomen": "CT Abdomen non-contrast: 7mm obstructing stone at left ureterovesical junction with moderate hydronephrosis", "urinalysis": "Urinalysis: RBC 50+, WBC 2, no bacteria, pH 5.5", "BMP": "Na 140, K 4.0, Cr 1.1, Ca 10.8 (upper normal)"}
SOAP_HISTORY_DB["Nephrolithiasis"] = {"HPI": "38M presents with sudden onset severe left flank pain radiating to groin that started 3 hours ago. Pain comes in waves, rates 10/10. Unable to sit still. Nausea with vomiting x2. Noticed blood in urine.", "ROS": {"GU": "flank pain, hematuria, groin pain", "GI": "nausea, vomiting"}, "Past_Medical_History": "2 prior kidney stones (passed spontaneously), gout, inadequate fluid intake", "Medications": "Allopurinol 100mg daily (poor compliance)", "Allergies": "NKDA", "Social_History": "Software developer, drinks minimal water, high protein diet, sedentary", "Physical_Examination": "Writhing in pain, unable to find comfortable position. CVA tenderness left. Abdomen soft, mild left lower quadrant tenderness. No peritoneal signs. Tachycardic."}

DISEASES_DB["Testicular Torsion"] = {"true_disease": "Testicular Torsion", "true_symptoms": ["sudden severe testicular pain", "nausea and vomiting", "absent cremasteric reflex", "high-riding testicle", "scrotal swelling"], "correct_treatment": "emergent surgical exploration and detorsion within 6 hours, attempt manual detorsion open book technique while awaiting OR, bilateral orchiopexy, doppler US if diagnosis uncertain", "lethal_treatments": ["antibiotics for presumed epididymitis without ruling out torsion"], "medical_history": "Adolescent or young adult, bell-clapper deformity", "difficulty": "medium", "critical_labs": ["doppler_US", "urinalysis"]}
VITALS_DB["Testicular Torsion"] = "HR 110, BP 140/85, RR 20, SpO2 99%, Temp 37.0C -- tachycardic from pain, afebrile (distinguishes from infection)"
LAB_RESULTS_DB["Testicular Torsion"] = {"doppler_US": "Scrotal Doppler US: absent blood flow to left testicle, testis rotated 540 degrees, edematous -- TORSION, requires emergent surgery", "urinalysis": "Urinalysis: normal -- no infection (helps distinguish from epididymitis)"}
SOAP_HISTORY_DB["Testicular Torsion"] = {"HPI": "16M presents with sudden onset severe left testicular pain that woke him from sleep 3 hours ago. Pain started without trauma. Associated nausea and vomiting x3. Pain is constant and worsening. No urinary symptoms.", "ROS": {"GU": "severe left testicular pain, swelling", "GI": "nausea, vomiting x3"}, "Past_Medical_History": "Previously healthy. No prior episodes. Not sexually active.", "Medications": "None", "Allergies": "NKDA", "Social_History": "High school student, athlete", "Physical_Examination": "In severe distress. Left testicle high-riding, horizontal lie, extremely tender. Absent cremasteric reflex on left. Negative Prehn sign (pain NOT relieved with elevation). Right testicle normal. No fever."}

DISEASES_DB["Pyelonephritis"] = {"true_disease": "Pyelonephritis", "true_symptoms": ["flank pain", "high fever", "dysuria", "nausea and vomiting", "CVA tenderness"], "correct_treatment": "IV ceftriaxone 1g or fluoroquinolone, IV fluids, blood cultures if septic, urine culture, admission if unable to tolerate PO or signs of sepsis", "lethal_treatments": ["oral antibiotics only if hemodynamically unstable"], "medical_history": "Recurrent UTIs, diabetes, kidney stones", "difficulty": "easy", "critical_labs": ["urinalysis", "CBC", "BMP", "blood_culture"]}
VITALS_DB["Pyelonephritis"] = "HR 108, BP 105/65, RR 20, SpO2 98%, Temp 39.5C -- febrile, tachycardic"
LAB_RESULTS_DB["Pyelonephritis"] = {"urinalysis": "Urinalysis: WBC 80+, bacteria many, nitrite positive, leukocyte esterase positive, WBC casts present -- upper tract infection", "CBC": "WBC 18.0 (left shift), Hgb 12.5, Plt 220", "BMP": "Na 138, K 3.8, Cr 1.3, Glucose 130", "blood_culture": "Blood cultures: Gram-negative rods at 12 hours -- E. coli"}
SOAP_HISTORY_DB["Pyelonephritis"] = {"HPI": "32F presents with 3 days of dysuria and frequency that progressed to right flank pain, high fever, and vomiting. Unable to keep fluids down. Had UTI symptoms that she tried to treat with cranberry juice.", "ROS": {"GU": "dysuria, frequency, flank pain, foul-smelling urine", "GI": "nausea, vomiting x4", "Neuro": "no confusion"}, "Past_Medical_History": "Recurrent UTIs (3 per year), Type 2 DM", "Medications": "Metformin 500mg BID", "Allergies": "Sulfa drugs (rash)", "Social_History": "Nurse, sexually active, uses diaphragm for contraception", "Physical_Examination": "Febrile, ill-appearing but alert. Right CVA tenderness on percussion. Mild suprapubic tenderness. No peritoneal signs. Well-hydrated."}

DISEASES_DB["Urinary Retention"] = {"true_disease": "Urinary Retention", "true_symptoms": ["inability to urinate", "suprapubic pain and fullness", "overflow incontinence", "lower abdominal distension", "restlessness"], "correct_treatment": "Foley catheter insertion with slow drainage max 500mL at a time to prevent decompression hematuria, post-void residual measurement, alpha-blocker tamsulosin, urology follow-up", "lethal_treatments": ["rapid complete bladder decompression over 1000mL at once"], "medical_history": "BPH, anticholinergic medications, post-operative", "difficulty": "easy", "critical_labs": ["BMP", "urinalysis", "bladder_US"]}
VITALS_DB["Urinary Retention"] = "HR 90, BP 155/90, RR 18, SpO2 98%, Temp 37.0C -- hypertensive from pain and distress"
LAB_RESULTS_DB["Urinary Retention"] = {"BMP": "Na 140, K 4.5, Cr 1.8 (mildly elevated -- obstructive), BUN 30", "urinalysis": "Urinalysis: WBC 5, RBC 10, no bacteria -- mild inflammation from distension", "bladder_US": "Bladder US: distended bladder volume approximately 1200mL -- acute urinary retention"}
SOAP_HISTORY_DB["Urinary Retention"] = {"HPI": "75M presents with 18 hours of inability to urinate despite strong urge. Progressive suprapubic pain and fullness. Dribbling small amounts. Was started on new cold medication (pseudoephedrine + diphenhydramine) 2 days ago.", "ROS": {"GU": "inability to void, suprapubic pain, dribbling", "GI": "mild lower abdominal pain"}, "Past_Medical_History": "BPH on tamsulosin (ran out 1 week ago), HTN, recently started OTC cold medication", "Medications": "Tamsulosin 0.4mg (stopped 1 week ago), lisinopril 10mg, pseudoephedrine/diphenhydramine (OTC cold medicine started 2 days ago)", "Allergies": "NKDA", "Social_History": "Retired, lives with wife, non-smoker", "Physical_Examination": "Uncomfortable, palpable distended bladder to umbilicus. Suprapubic tenderness. DRE: enlarged smooth prostate, no nodules. No CVA tenderness."}

# ---------------------------------------------------------------------------
# CLASS 10: ENVIRONMENTAL / IMMUNOLOGIC (5 diseases)
# ---------------------------------------------------------------------------

DISEASES_DB["Anaphylaxis"] = {"true_disease": "Anaphylaxis", "true_symptoms": ["urticaria and angioedema", "wheezing and stridor", "hypotension", "abdominal cramping", "sense of impending doom"], "correct_treatment": "IM epinephrine 0.3mg anterolateral thigh repeat every 5-15 minutes, IV fluids wide open, albuterol for bronchospasm, IV diphenhydramine, IV methylprednisolone, monitor for biphasic reaction", "lethal_treatments": ["IV epinephrine bolus (cardiac arrest risk)", "relying on antihistamines alone"], "medical_history": "Known allergies, prior anaphylaxis, bee sting", "difficulty": "medium", "critical_labs": ["tryptase", "CBC"]}
VITALS_DB["Anaphylaxis"] = "HR 140, BP 65/30, RR 30, SpO2 85%, Temp 37.0C -- anaphylactic shock, severe hypotension and hypoxia"
LAB_RESULTS_DB["Anaphylaxis"] = {"tryptase": "Serum tryptase: 45 ng/mL (CRITICAL HIGH -- confirms mast cell degranulation/anaphylaxis)", "CBC": "WBC 8.0, Hgb 14.0, Plt 220 -- normal"}
SOAP_HISTORY_DB["Anaphylaxis"] = {"HPI": "28F presents with sudden onset diffuse hives, lip and tongue swelling, wheezing, and lightheadedness 15 minutes after eating shrimp at restaurant. Rapidly progressive. Has known shellfish allergy but did not know dish contained shrimp.", "ROS": {"Derm": "diffuse urticaria, facial swelling", "Resp": "wheezing, throat tightness, stridor", "CV": "lightheadedness, palpitations", "GI": "abdominal cramping"}, "Past_Medical_History": "Shellfish allergy (prior mild reaction -- hives only), asthma, carries EpiPen (expired, did not use)", "Medications": "Albuterol PRN, expired EpiPen (did not administer)", "Allergies": "Shellfish (anaphylaxis), penicillin (rash)", "Social_History": "Teacher, non-smoker", "Physical_Examination": "Diffuse urticaria. Angioedema of lips and tongue. Audible stridor and wheezing. Hypotensive. Tachycardic. Using accessory muscles. Abdomen with diffuse tenderness."}

DISEASES_DB["Heat Stroke"] = {"true_disease": "Heat Stroke", "true_symptoms": ["core temperature above 40C", "altered mental status", "hot dry skin", "tachycardia", "seizures"], "correct_treatment": "rapid cooling ice water immersion or evaporative cooling, cold IV fluids, benzodiazepines for shivering or seizures, intubation if GCS below 8, monitor for rhabdomyolysis and DIC", "lethal_treatments": ["antipyretics acetaminophen or NSAIDs (ineffective and hepatotoxic in heat stroke)", "delaying cooling for workup"], "medical_history": "Exertion in heat, elderly in hot environment", "difficulty": "medium", "critical_labs": ["BMP", "CBC", "CK", "coagulation"]}
VITALS_DB["Heat Stroke"] = "HR 145, BP 90/55, RR 30, SpO2 95%, Temp 42.1C -- CRITICAL hyperthermia, tachycardic, hypotensive"
LAB_RESULTS_DB["Heat Stroke"] = {"BMP": "Na 148 (HIGH -- dehydration), K 5.5 (HIGH), Cr 2.5, Glucose 65", "CBC": "WBC 18.0, Hgb 17.0 (hemoconcentration), Plt 80 (DIC developing)", "CK": "CK: 25000 U/L (CRITICAL -- rhabdomyolysis)", "coagulation": "PT 22, INR 2.5, aPTT 55, fibrinogen 100 -- DIC"}
SOAP_HISTORY_DB["Heat Stroke"] = {"HPI": "22M military recruit collapsed during 10-mile training run in 38C heat. Found confused and combative. Core temp 42.1C per rectal thermometer. Hot dry skin. Witnessed seizure in field.", "ROS": {"Neuro": "confusion, combative, seizure witnessed", "Derm": "hot dry skin, no sweating"}, "Past_Medical_History": "Previously healthy, new recruit in basic training x 2 weeks", "Medications": "None", "Allergies": "NKDA", "Social_History": "Military recruit, recently moved from cold climate, not heat-acclimatized, was not adequately hydrating", "Physical_Examination": "Combative, confused, GCS 10. Core temp 42.1C. Skin hot and dry (anhidrosis). Tachycardic. Hypotensive. No focal neurological deficits. Dark urine (myoglobinuria)."}

DISEASES_DB["Severe Hypothermia"] = {"true_disease": "Severe Hypothermia", "true_symptoms": ["core temperature below 30C", "altered consciousness", "bradycardia", "J waves on ECG", "muscle rigidity"], "correct_treatment": "active core rewarming with warm IV fluids 40-42C, warm humidified oxygen, bear hugger, avoid rough handling (risk of VFib), cardiac monitoring, ECMO if cardiac arrest", "lethal_treatments": ["rapid surface rewarming alone", "pronouncing death before rewarming -- you are not dead until warm and dead"], "medical_history": "Environmental exposure, homeless, elderly", "difficulty": "hard", "critical_labs": ["BMP", "ECG", "ABG", "CBC"]}
VITALS_DB["Severe Hypothermia"] = "HR 32, BP 75/45, RR 6, SpO2 88%, Temp 27.5C -- severe bradycardia, profound hypothermia"
LAB_RESULTS_DB["Severe Hypothermia"] = {"BMP": "Na 140, K 3.0, Glucose 50 (LOW), Cr 1.5", "ECG": "Marked sinus bradycardia rate 32, Osborn (J) waves in precordial leads, prolonged QT -- classic hypothermia", "ABG": "pH 7.22, pCO2 50, pO2 55 -- mixed acidosis (temperature corrected)", "CBC": "WBC 4.0, Hgb 16.0 (hemoconcentration), Plt 90"}
SOAP_HISTORY_DB["Severe Hypothermia"] = {"HPI": "Homeless 60M found unresponsive outdoors by police on a night with ambient temperature -5C. Unknown down time. Minimally responsive. Cold and rigid. Bystanders report he was seen drinking earlier.", "ROS": {"Neuro": "unresponsive"}, "Past_Medical_History": "Unknown -- homeless, no medical records available. Smells of alcohol.", "Medications": "Unknown", "Allergies": "Unknown", "Social_History": "Homeless, known to frequent shelters, alcohol use suspected", "Physical_Examination": "Unresponsive, GCS 5. Core temp 27.5C. Rigid musculature. Bradycardic, weak pulse. Pupils sluggish. Cold skin. No visible trauma."}

DISEASES_DB["Snakebite Envenomation"] = {"true_disease": "Snakebite Envenomation", "true_symptoms": ["fang marks with local swelling", "progressive edema", "ecchymosis", "metallic taste", "coagulopathy"], "correct_treatment": "CroFab antivenom 4-6 vials IV initial dose, repeat if swelling progresses, mark advancing edge of swelling, IV fluids, tetanus prophylaxis, avoid tourniquets and incision", "lethal_treatments": ["tourniquet", "incision and suction", "ice to wound"], "medical_history": "Outdoor exposure, rural area", "difficulty": "medium", "critical_labs": ["CBC", "coagulation", "BMP", "fibrinogen"]}
VITALS_DB["Snakebite Envenomation"] = "HR 115, BP 95/60, RR 22, SpO2 97%, Temp 37.5C -- tachycardic, mildly hypotensive"
LAB_RESULTS_DB["Snakebite Envenomation"] = {"CBC": "WBC 15.0, Hgb 12.0, Plt 45 (CRITICAL LOW -- venom-induced thrombocytopenia)", "coagulation": "PT 35, INR 4.5 (CRITICAL), aPTT 85 -- severe coagulopathy from venom", "BMP": "Na 138, K 4.8, Cr 1.5, CK 2500 (myotoxicity)", "fibrinogen": "Fibrinogen: 50 mg/dL (CRITICAL LOW -- consumptive coagulopathy)"}
SOAP_HISTORY_DB["Snakebite Envenomation"] = {"HPI": "35M presents 2 hours after being bitten on right hand by a rattlesnake while hiking. Progressive swelling from hand to forearm. Noted 2 puncture wounds. Metallic taste in mouth. Mild nausea. Brought dead snake for identification.", "ROS": {"Derm": "progressive swelling right hand and forearm, ecchymosis", "GI": "nausea, metallic taste", "Neuro": "tingling around mouth"}, "Past_Medical_History": "Previously healthy, no prior snakebites", "Medications": "None", "Allergies": "NKDA", "Social_History": "Avid hiker, lives in rural Arizona, was hiking alone", "Physical_Examination": "Two puncture wounds on right dorsal hand. Edema extending from hand to mid-forearm (marked at 2cm proximal progression per 15 minutes). Ecchymosis developing. Tender throughout. Distal pulses intact. Mild perioral paresthesias."}

DISEASES_DB["Angioedema"] = {"true_disease": "Angioedema", "true_symptoms": ["rapid swelling of face lips tongue", "difficulty breathing", "stridor", "no urticaria if hereditary", "abdominal pain"], "correct_treatment": "if ACE inhibitor-induced: discontinue ACE inhibitor, icatibant or C1 esterase inhibitor concentrate, prepare for intubation or surgical airway; if allergic: epinephrine and antihistamines", "lethal_treatments": ["continued ACE inhibitor use", "relying on epinephrine alone for ACE inhibitor angioedema (may not respond)"], "medical_history": "ACE inhibitor use, hereditary angioedema", "difficulty": "hard", "critical_labs": ["CBC", "C4_level", "tryptase"]}
VITALS_DB["Angioedema"] = "HR 95, BP 150/90, RR 24, SpO2 93%, Temp 37.0C -- hypertensive (on ACE inhibitor), hypoxic from airway compromise"
LAB_RESULTS_DB["Angioedema"] = {"CBC": "WBC 8.0, Hgb 14.0, Plt 220 -- normal", "C4_level": "C4: 8 mg/dL (LOW -- suggests bradykinin-mediated, not histamine)", "tryptase": "Serum tryptase: 5 ng/mL (normal -- NOT allergic/mast cell mediated, confirms ACE inhibitor cause)"}
SOAP_HISTORY_DB["Angioedema"] = {"HPI": "65M presents with 4 hours of progressive swelling of tongue and lips. Now having difficulty speaking and swallowing. Mild stridor noted. He has been on lisinopril for 8 years without prior issues. No urticaria. No known allergen exposure.", "ROS": {"ENT": "tongue and lip swelling, difficulty swallowing, voice change", "Resp": "mild stridor, dyspnea", "Derm": "NO urticaria (important -- suggests bradykinin not histamine)"}, "Past_Medical_History": "HTN on lisinopril x 8 years, Type 2 DM", "Medications": "Lisinopril 20mg daily, metformin 1000mg BID", "Allergies": "NKDA", "Social_History": "Retired engineer, African American (higher risk for ACE inhibitor angioedema), non-smoker", "Physical_Examination": "Significant tongue and lip edema. Voice muffled. Mild inspiratory stridor. No urticaria anywhere. Oropharynx: tongue filling oral cavity, uvula edematous. Lungs clear. Airway assessment: concerning for progression."}
