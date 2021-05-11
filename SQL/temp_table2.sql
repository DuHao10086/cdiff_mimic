DROP MATERIALIZED VIEW IF EXISTS first_lab_temp2 CASCADE;
CREATE MATERIALIZED VIEW first_lab_temp2 as


select ft.subject_id, ft.hadm_id, ft.icustay_id,
ft.outtime, ft.deathtime, ft.los,
ft.admit_age, ft.oasis, ft.gender, 
ft.if_vasopressor, ft.if_mechvent, ft.vasonum, 
ft.first_careunit, ft.last_careunit,
-- le.charttime, le.itemid, le.value, le.valuenum, le.valueuom, -- dlab.label, 
ft.mort_icu,
-- CASE when a.deathtime between a.admittime and a.dischtime THEN 1 ELSE 0 END AS mort_hosp, 
ft.icu_rank, 
-- case    when le.itemid = 50868 then 'ANION GAP'
--         when le.itemid = 50862 then 'ALBUMIN'
--         when le.itemid = 50882 then 'BICARBONATE'
--         when le.itemid = 50885 then 'BILIRUBIN'
--         when le.itemid = 50912 then 'CREATININE'
--         when le.itemid = 50806 then 'CHLORIDE'
--         when le.itemid = 50902 then 'CHLORIDE'
--         when le.itemid = 50809 then 'GLUCOSE'
--         when le.itemid = 50931 then 'GLUCOSE'
--         when le.itemid = 50810 then 'HEMATOCRIT'
--         when le.itemid = 51221 then 'HEMATOCRIT'
--         when le.itemid = 50811 then 'HEMOGLOBIN'
--         when le.itemid = 51222 then 'HEMOGLOBIN'
--         when le.itemid = 50813 then 'LACTATE'
--         when le.itemid = 50960 then 'MAGNESIUM'
--         when le.itemid = 50970 then 'PHOSPHATE'
--         when le.itemid = 51265 then 'PLATELET'
--         when le.itemid = 50822 then 'POTASSIUM'
--         when le.itemid = 50971 then 'POTASSIUM'
--         when le.itemid = 51275 then 'PTT'
--         when le.itemid = 51237 then 'INR'
--         when le.itemid = 51274 then 'PT'
--         when le.itemid = 50824 then 'SODIUM'
--         when le.itemid = 50983 then 'SODIUM'
--         when le.itemid = 51006 then 'BUN'
--         when le.itemid = 51300 then 'WBC'
--         when le.itemid = 51301 then 'WBC'
--         -- Calcium
--         when le.itemid = 50893 then 'CALCIUM'
--         -- Free calcium
--         when le.itemid = 50808 then 'FREECALCIUM'
-- 		ELSE NULL
--         end AS lablabel


from first_lab_temp ft

left join mimiciii.labevents le 
on ft.subject_id = le.subject_id
-- WHERE
-- le.charttime >= (ft.intime - interval '24 hour' ) 
-- AND le.charttime <= (ft.outtime + interval '24 hour' ) 
-- AND le.valuenum IS NOT null
-- AND le.valuenum > 0;