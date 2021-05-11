DROP MATERIALIZED VIEW IF EXISTS first_lab_temp CASCADE;
CREATE MATERIALIZED VIEW first_lab_temp as

select icu.subject_id, icu.hadm_id, icu.icustay_id, 
    icu.outtime, icu.intime, a.deathtime, icu.los,
	ROUND((cast(a.admittime as date) - cast(p.dob as date))/365.24, 2) as admit_age, os.oasis, 
	p.gender, 
--if the use of vasopressor is true
case when vp.endtime  >= (icu.intime - interval '24 hour') or vp.starttime <= (icu.outtime + interval '24 hour') then 'TRUE'
else null 
end as if_vasopressor, 
--if the use of mechanical ventilation is true
case when vs.mechvent = 1.0 --and (vd.endtime  >= (icu.intime - interval '24 hour') or vd.starttime <= (icu.outtime + interval '24 hour'))
then 1 else 0 end as if_mechvent,
vp.vasonum,

icu.first_careunit, icu.last_careunit,
-- le.charttime, le.itemid, le.value, le.valuenum, le.valueuom, -- dlab.label, 

CASE when a.deathtime between icu.intime and icu.outtime THEN 1 ELSE 0 END AS mort_icu,
-- CASE when a.deathtime between a.admittime and a.dischtime THEN 1 ELSE 0 END AS mort_hosp, 

RANK() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime) as icu_rank

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
	
from mimiciii.icustays icu
	
left join mimiciii.patients p 
on icu.subject_id = p.subject_id

left join mimiciii.admissions a 
on icu.subject_id = a.subject_id 
AND icu.hadm_id = a.hadm_id
	
-- left join mimiciii.labevents le 
-- on icu.subject_id = le.subject_id

-- AND le.charttime >= (icu.intime - interval '24 hour' ) AND le.charttime <= (icu.outtime + interval '24 hour' )
-- -- sanity check for lab values
-- AND le.valuenum IS NOT null AND le.valuenum > 0 -- lab values cannot be 0 and cannot be negative

--inner join d_labitems dlab on dlab.itemid = le.itemid
	
left join public.oasis os 
on icu.icustay_id = os.icustay_id --AND icu.hadm_id = os.hadm_id

left join public.vasopressordurations vp on 
icu.icustay_id = vp.icustay_id

left join public.ventsettings vs 
on icu.icustay_id = vs.icustay_id
--left join ventdurations vd on vd.icustay_id = icu.icustay_id
	
-- WHERE
-- le.charttime >= (icu.intime - interval '24 hour' ) 
-- AND le.charttime <= (icu.outtime + interval '24 hour' ) 
-- AND le.valuenum IS NOT null
-- AND le.valuenum > 0