# AI Full Pipeline - Battery AI Co-Scientist

Tai lieu nay mo ta luong AI muc tieu khi he thong duoc implement day du, khong chi dung o MVP. MVP hien tai da co duong chay `data -> model -> recommendation -> dashboard`; ban full se mo rong thanh vong lap AI co-scientist khep kin: thu thap du lieu, hieu ngu canh khoa hoc, toi uu cong thuc, de xuat thi nghiem, ghi nhan ket qua moi, roi tu hoc lai.

## 1. Muc tieu he thong

He thong dong vai tro AI co-scientist cho vat lieu anode Si/MXene/alginate:

1. Tong hop du lieu tu quy trinh che tao, SEM, EDX, dien hoa, tai lieu khoa hoc va ket qua thi nghiem moi.
2. Tao mot master dataset co kiem tra chat luong va truy vet nguon.
3. Trich xuat dac trung vat lieu tu anh SEM va du lieu thanh phan.
4. Huan luyen cac model du doan hieu nang dien hoa, dac biet `retention_100`.
5. Toi uu cong thuc bang uncertainty-aware Bayesian Optimization.
6. De xuat thi nghiem tiep theo kem ly do khoa hoc, do tin cay, rui ro va dieu kien che tao.
7. Nhan phan hoi cua nha nghien cuu, cap nhat ket qua moi, va lap lai vong active learning.

## 2. So do tong quan

```text
Raw lab data + SEM images + papers + human notes
        |
        v
Data ingestion and validation
        |
        v
Feature extraction
  - process features
  - SEM morphology/crack features
  - EDX composition features
  - electrochemical targets
  - literature-derived priors
        |
        v
Master experiment store
        |
        v
Model training and evaluation
  - baseline predictor
  - uncertainty surrogate
  - feature importance/explainability
        |
        v
Candidate generation
  - chemically valid recipe grid
  - constraint filtering
  - acquisition scoring
        |
        v
AI scientist reasoning layer
  - compare against prior experiments
  - retrieve relevant papers
  - explain mechanism and risks
  - produce next-experiment plan
        |
        v
Human review
  - accept
  - modify
  - reject
        |
        v
Lab execution and new results
        |
        v
Feedback loop: update dataset, retrain, re-rank
```

## 3. Data layer

### 3.1 Inputs

| Nguon | Noi dung | Vi tri hien tai |
|---|---|---|
| Process data | Si, MXene, alginate, carbon, mixing time, drying temp, pressing pressure | `data/raw/process_data.csv` |
| SEM tabular features | particle size, porosity, agglomeration, crack density, uniformity | `data/raw/sem_features.csv` |
| SEM images | Anh SEM goc de phan tich morphology va crack | `data/raw/sem_images/` |
| EDX data | Si, Ti, C, O, impurity percent | `data/raw/edx_data.csv` |
| Electrochemistry | initial capacity, capacity_50, capacity_100, retention_100, CE, Rct | `data/raw/electrochem_data.csv` |
| Literature | Paper, mechanism, known constraints, recommended ranges | `literature_notes.md` |
| Human decisions | Accept/Modify/Reject, lab notes, failed runs | Can co DB khi full |

### 3.2 Validation

Full implementation phai validate truoc khi model nhin thay du lieu:

- Moi bang thuc nghiem phai co `sample_id` duy nhat.
- Cac bang process, SEM, EDX, electrochem phai match cung tap `sample_id`.
- Tong thanh phan Si + MXene + alginate + carbon phai bang 100 wt%.
- Target `retention_100` khong duoc null; neu thieu co the tinh tu `capacity_100 / initial_capacity * 100`.
- Feature dung de recommend khong duoc leak thong tin chi co sau khi thi nghiem.
- Moi ban ghi can co metadata: ngay lam, operator, batch, instrument, raw-file lineage.

Hien tai logic validation nam chinh trong `src/build_dataset.py`.

## 4. Feature engineering

### 4.1 Process features

Dung truc tiep cac dieu kien che tao:

- `si_content`
- `mxene_content`
- `alginate_content`
- `carbon_content`
- `mixing_time`
- `drying_temp`
- `pressing_pressure`

### 4.2 SEM image features

Luong full:

```text
SEM image
  -> decode image
  -> remove footer/instrument annotation
  -> optional contrast enhancement
  -> material segmentation
  -> crack detection
  -> morphology metrics
  -> quality flag
```

MVP hien tai da co:

- Otsu/morphology segmentation fallback.
- Optional SAM checkpoint neu co model.
- Crack metrics: area fraction, count, length density, mean width.
- Particle area fraction va overlay de review.

Khi full, nen mo rong:

- Fine-tuned SAM/SAM2 hoac micro-sam cho anh SEM domain-specific.
- QC score cho anh mo, scale bar loi, footer cat sai, anh khong dung domain.
- Luu mask/overlay vao artifact store de audit.

### 4.3 EDX and engineered features

Feature hien tai:

- `si_ti_ratio`
- `c_o_ratio`
- `si_mxene_ratio`
- `capacity_fade`

Full implementation nen them:

- normalized elemental ratios theo batch.
- impurity risk score.
- interaction terms giua process va morphology.
- literature-derived priors ve range an toan.

## 5. Master experiment store

MVP hien luu `data/processed/master_dataset.csv`.

Ban full nen thay bang database/experiment registry:

```text
experiments
samples
process_conditions
sem_measurements
edx_measurements
electrochem_results
model_runs
recommendations
human_decisions
artifact_files
literature_sources
```

CSV van co the export de training, nhung source of truth nen la DB de truy vet:

- Mau nao tao model nao.
- Model nao tao recommendation nao.
- Recommendation nao duoc chap nhan va cho ket qua thuc nghiem nao.

## 6. Model layer

### 6.1 Baseline predictor

Model baseline hien tai:

- `RandomForestRegressor`
- target: `retention_100`
- artifact: `outputs/trained_model.pkl`
- metrics: `outputs/model_metrics.json`
- feature importance: `outputs/feature_importance.csv`

Vai tro:

- Du doan nhanh hieu nang.
- Cho feature importance de giai thich so bo.
- Lam baseline de so sanh voi model phuc tap hon.

### 6.2 Uncertainty surrogate

Model surrogate hien tai:

- `GaussianProcessRegressor`
- `StandardScaler`
- acquisition: UCB beta = 1.0

Vai tro:

- Du doan mean retention.
- Tinh uncertainty `std`.
- Uu tien diem co tiem nang cao hoac vung chua duoc explore.

### 6.3 Full model stack

Khi implement day du, model layer nen co:

- Baseline tabular model: RandomForest/ExtraTrees/XGBoost neu du lieu du lon.
- Uncertainty model: Gaussian Process, ensemble, hoac conformal prediction.
- Multi-objective optimization: retention, capacity, CE, Rct, stability, cost, safety.
- Model registry: version, training data hash, metrics, feature schema.
- Drift detection: phat hien batch moi khac distribution cu.
- Calibration: predicted uncertainty phai match sai so thuc nghiem.

## 7. Candidate generation and optimization

### 7.1 Search space

Hien tai `src/recommend.py` tao luoi roi loc:

- Si: 55-75
- MXene: 10-30
- Alginate: 5-15
- Drying temp: 70-100
- Mixing time: 30-60
- Carbon = 100 - Si - MXene - alginate
- Carbon phai >= 5
- Cong thuc da co trong master dataset bi loai.

### 7.2 Ranking

Hien tai:

```text
acquisition_score = predicted_retention_mean + beta * predicted_retention_std
```

Output:

- rank
- recipe
- predicted retention
- uncertainty
- acquisition score
- confidence
- reason

### 7.3 Full optimization

Ban full nen mo rong:

- Constraint-aware optimizer thay cho fixed grid khi search space lon.
- Multi-objective acquisition.
- Hard constraints: tong wt%, carbon min, equipment limits.
- Soft constraints: cost, reproducibility, risk, literature prior.
- Batch recommendation: de xuat 3-10 mau cung luc nhung da dang, khong trung quanh mot diem.
- Exploration budget: neu lab chi lam duoc N mau, model phai toi uu theo budget.

## 8. AI scientist reasoning layer

Day la phan khac biet giua MVP va full implementation.

Luong full:

```text
Top candidate recipes
  + nearest historical experiments
  + model uncertainty
  + feature importance
  + retrieved papers
  + lab constraints
        |
        v
LLM/RAG scientist
        |
        v
Structured recommendation report
```

### 8.1 Retrieval

RAG index nen gom:

- `literature_notes.md`
- paper PDFs hoac abstracts.
- previous experiment notes.
- failed experiments.
- synthesis constraints.
- safety/process SOPs.

Retrieval output phai co citation/source id, khong chi tra loi tu tri nho model.

### 8.2 Reasoning output

Moi recommendation day du nen co:

- Cong thuc va dieu kien che tao.
- Model prediction va uncertainty.
- Ly do khoa hoc: vai tro Si/MXene/alginate/carbon.
- So sanh voi mau gan nhat da thu.
- Rui ro: crack, agglomeration, conductivity, binder imbalance.
- De xuat kiem tra SEM/EDX/electrochem sau khi che tao.
- Dieu kien dung/khong dung.
- Citation tu literature hoac prior experiment.

### 8.3 Guardrails

LLM layer khong duoc tu y tao so lieu:

- So lieu model phai lay tu artifacts/DB.
- Citation phai den tu retrieval.
- Neu thieu du lieu, phai noi `insufficient evidence`.
- Output nen la JSON/schema truoc, sau do render thanh report.

## 9. Human-in-the-loop

Dashboard full khong chi hien placeholder Accept/Modify/Reject. No phai ghi lai quyet dinh:

```text
recommendation_id
decision: accept | modify | reject
modified_recipe
reason_from_scientist
operator
timestamp
follow_up_experiment_id
```

Neu accept:

1. Tao experiment plan.
2. Gan batch/run id.
3. Export protocol cho lab.
4. Cho upload ket qua moi.

Neu modify:

1. Ghi recipe ban dau va ban sua.
2. Ghi ly do sua.
3. Chay lai feasibility/prediction.

Neu reject:

1. Ghi ly do reject.
2. Dung ly do nay lam feedback cho constraint/reward model.

## 10. Closed-loop learning

Luong active learning day du:

```text
1. Model tao recommendation.
2. Scientist review va chon thi nghiem.
3. Lab che tao mau.
4. SEM/EDX/electrochem duoc ingest vao system.
5. Pipeline validate va tinh feature.
6. Ket qua moi duoc them vao experiment store.
7. Model retrain hoac incremental update.
8. Dashboard so sanh prediction vs actual.
9. Optimizer cap nhat vung search tiep theo.
```

Metric quan trong:

- prediction error tren mau moi.
- uncertainty calibration.
- hit rate cua top-N recommendation.
- so thi nghiem can de cai thien target.
- ty le recommendation bi reject/modify.

## 11. Observability and governance

Full implementation can co:

- Data quality report moi lan ingest.
- Model card moi lan train.
- Experiment lineage.
- Cost/latency cho LLM/RAG neu dung external API.
- Prompt/version tracking.
- Audit log cho human decisions.
- Security: khong day raw proprietary data ra model ngoai neu chua duoc phe duyet.
- Backup artifacts: raw data, masks, model bundle, metrics, recommendation report.

## 12. Current MVP mapping

| Phan | Hien co | Full implementation can them |
|---|---|---|
| End-to-end command | `python run_pipeline.py` | scheduler, run registry, failure recovery |
| Data ingestion | CSV + synthetic fallback | DB, upload/API, instrument integration |
| Validation | sample_id, composition, missing values | lineage, schema versioning, QC reports |
| SEM analysis | threshold/SAM fallback, crack metrics | fine-tuned SAM/SAM2, mask store, QC |
| Model | RF + GP surrogate | registry, multi-objective models, calibration |
| Recommendation | top-3 GP/UCB grid search | constraint optimizer, batch BO, lab budget |
| AI reasoning | simple textual reason | LLM/RAG scientist with citations |
| Dashboard | view data/model/recs, decision placeholders | persisted decisions and experiment planning |
| Feedback loop | manual rerun | closed-loop retrain from new lab results |

## 13. Full pipeline acceptance criteria

He thong chi duoc xem la full implementation khi dat cac tieu chi sau:

- Co the ingest du lieu thi nghiem that, khong phu thuoc synthetic fallback.
- Moi sample co lineage tu raw file den feature den model training row.
- Recommendation duoc luu thanh ban ghi co id va version model.
- Scientist co the accept/modify/reject va decision duoc persist.
- Ket qua thi nghiem moi co the nap lai de retrain/re-rank.
- LLM/RAG neu bat len phai co citation, schema output va guardrail chong fabricate so lieu.
- Dashboard hien prediction vs actual cho cac vong lap truoc.
- Model registry luu duoc artifact, metrics, data hash va feature schema.
- Co test/validation cho data quality, recommendation constraints va report schema.

## 14. One-command target

MVP hien tai:

```bash
python run_pipeline.py
```

Full target nen co cac mode ro rang:

```bash
python run_pipeline.py ingest
python run_pipeline.py train
python run_pipeline.py recommend
python run_pipeline.py report
python run_pipeline.py close-loop --experiment-id EXP001
```

Trong do `report` tao AI scientist report, con `close-loop` ingest ket qua moi roi cap nhat model va recommendation.
