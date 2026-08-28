# Code Pipeline Plan — AI Co-Scientist cho điện cực âm nano-silicon/few-layer Ti₃C₂Tₓ MXene

**Topic đề tài:**  
**Tối ưu hóa quy trình chế tạo điện cực âm từ hệ nano-silicon/few-layer Ti₃C₂Tₓ MXene sử dụng binder sodium alginate cho pin lithium-ion bằng trí tuệ nhân tạo đa phương thức**

**English title:**  
**Multimodal AI-Assisted Optimization of Nano-Silicon/Few-Layer Ti₃C₂Tₓ MXene Anode Fabrication Using Sodium Alginate Binder for Lithium-Ion Batteries**

---

## 1. Mục tiêu MVP code

Mục tiêu của agent là build một prototype chạy được, không cần hoàn chỉnh như hệ thống research thật.

Pipeline MVP:

```text
Input CSV data
↓
Build master dataset
↓
Train model dự đoán cycling stability / capacity retention
↓
Recommend top 3 điều kiện chế tạo tiếp theo
↓
Hiển thị trên Streamlit dashboard
```

Target chính:

```text
retention_100 hoặc retention_50
```

Nếu chưa có dữ liệu thật, dùng synthetic data trước, nhưng schema phải giống dữ liệu thật sau này.

---

## 2. Nguyên tắc triển khai

Không build quá nặng ở MVP.

Nên làm trước:

```text
- Synthetic/sample dataset
- Data schema rõ ràng
- Master dataset builder
- Baseline model
- Recommendation engine
- Dashboard demo
- Unit tests đơn giản
```

Để sau:

```text
- LLM literature mining thật
- SAM/SAM2 segmentation thật
- Full Gaussian Process + Bayesian Optimization
- Database backend
- Auto-update model sau mỗi thí nghiệm thật
- User authentication
```

Cách giải thích MVP khi trình bày:

> Prototype này mô phỏng pipeline AI Co-scientist. Hệ thống gom dữ liệu chế tạo, SEM, EDX và điện hóa thành một master dataset. Model học mối liên hệ giữa điều kiện chế tạo và độ bền chu kỳ, sau đó đề xuất top 3 điều kiện chế tạo tiếp theo. Trong nghiên cứu thật, dữ liệu synthetic sẽ được thay bằng dữ liệu thí nghiệm thực tế.

---

# 3. Phase code chi tiết

---

## Phase 0 — Setup project structure

### Mục tiêu

Tạo repo code sạch, dễ mở rộng.

### Folder structure

```text
battery-ai-coscientist/
│
├── data/
│   ├── raw/
│   │   ├── process_data.csv
│   │   ├── sem_features.csv
│   │   ├── edx_data.csv
│   │   ├── electrochem_data.csv
│   │   └── sem_images/
│   │
│   └── processed/
│       └── master_dataset.csv
│
├── src/
│   ├── build_dataset.py
│   ├── train_model.py
│   ├── recommend.py
│   ├── sem_features.py
│   ├── edx_features.py
│   └── utils.py
│
├── app/
│   └── streamlit_app.py
│
├── outputs/
│   ├── trained_model.pkl
│   ├── model_metrics.json
│   ├── feature_importance.csv
│   └── recommendations.csv
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_recommend.py
│
├── run_pipeline.py
├── requirements.txt
└── README.md
```

### Acceptance test

Chỉ sang phase sau khi đạt:

```text
[ ] Repo có đủ folder.
[ ] Có requirements.txt.
[ ] Có README hướng dẫn chạy.
[ ] Không bị lỗi path khi chạy từ root project.
```

---

## Phase 1 — Tạo data schema mẫu

### Mục tiêu

Tạo dữ liệu mẫu đúng format để pipeline chạy được. Nếu chưa có dữ liệu thật thì dùng synthetic data trước.

---

### File 1: `process_data.csv`

Dữ liệu điều kiện chế tạo.

```csv
sample_id,si_content,mxene_content,alginate_content,carbon_content,mixing_time,drying_temp,pressing_pressure
S001,60,20,10,10,30,80,5
S002,65,15,10,10,45,80,5
S003,70,10,10,10,30,90,6
```

Ý nghĩa:

```text
si_content = hàm lượng nano-silicon
mxene_content = hàm lượng MXene
alginate_content = hàm lượng binder sodium alginate
carbon_content = hàm lượng carbon dẫn điện nếu có
mixing_time = thời gian trộn slurry
乾ing_temp / drying_temp = nhiệt độ sấy điện cực
pressing_pressure = áp lực ép điện cực nếu có
```

---

### File 2: `sem_features.csv`

Dữ liệu feature từ ảnh SEM.

```csv
sample_id,particle_size_mean,porosity_score,agglomeration_index,crack_density,surface_uniformity
S001,120,0.42,0.31,0.08,0.75
S002,95,0.48,0.25,0.05,0.82
S003,160,0.35,0.45,0.12,0.64
```

Feature cần có:

```text
particle_size_mean
porosity_score
agglomeration_index
crack_density
surface_uniformity
```

MVP có thể tạo synthetic SEM feature trước. Sau đó mới thay bằng feature extract từ ảnh SEM thật.

---

### File 3: `edx_data.csv`

Dữ liệu thành phần nguyên tố từ EDX.

```csv
sample_id,si_percent,ti_percent,c_percent,o_percent,impurity_percent
S001,42,18,30,9,1
S002,45,15,31,8,1
S003,48,12,28,10,2
```

Feature có thể tạo thêm:

```text
si_ti_ratio
c_o_ratio
impurity_score
```

---

### File 4: `electrochem_data.csv`

Dữ liệu hiệu suất điện hóa. Đây là label/target.

```csv
sample_id,initial_capacity,capacity_50,capacity_100,retention_100,coulombic_efficiency,rct
S001,1200,920,850,70.8,97.5,120
S002,1150,980,920,80.0,98.2,90
S003,1300,850,760,58.5,96.8,160
```

Target chính:

```text
retention_100
```

Nếu chưa có 100 cycles:

```text
retention_50
```

hoặc:

```text
capacity_fade_rate
```

### Acceptance test

```text
[ ] Mỗi file có sample_id.
[ ] Có ít nhất 30–50 dòng synthetic data.
[ ] Không có missing value ở target.
[ ] Các cột số đọc được bằng pandas.
[ ] Có data_dictionary.md giải thích từng cột.
```

---

## Phase 2 — Build master dataset

### Mục tiêu

Merge các nguồn dữ liệu thành một bảng duy nhất.

### File cần code

```text
src/build_dataset.py
```

### Chức năng

```text
1. Load process_data.csv
2. Load sem_features.csv
3. Load edx_data.csv
4. Load electrochem_data.csv
5. Merge theo sample_id
6. Tạo feature mới
7. Save master_dataset.csv
```

### Feature engineering cơ bản

```text
si_mxene_ratio = si_content / mxene_content
si_ti_ratio = si_percent / ti_percent
capacity_fade = initial_capacity - capacity_100
retention_100 = capacity_100 / initial_capacity * 100, nếu file chưa có retention_100
```

### Output

```text
data/processed/master_dataset.csv
```

### Acceptance test

```text
[ ] master_dataset.csv được tạo.
[ ] Không mất sample_id sau khi merge.
[ ] Số dòng master_dataset đúng với số sample hợp lệ.
[ ] Không có NaN ở feature chính.
[ ] Có target retention_100.
[ ] File tests/test_dataset.py pass.
```

---

## Phase 3 — Train baseline model

### Mục tiêu

Train model dự đoán độ bền chu kỳ của điện cực.

### File cần code

```text
src/train_model.py
```

### Model MVP

Dùng trước:

```text
RandomForestRegressor
```

hoặc:

```text
XGBoostRegressor nếu setup được nhanh
```

### Input X

```text
process features + SEM features + EDX features
```

Ví dụ:

```text
si_content
mxene_content
alginate_content
carbon_content
mixing_time
drying_temp
pressing_pressure
particle_size_mean
porosity_score
agglomeration_index
crack_density
surface_uniformity
si_ti_ratio
c_o_ratio
```

### Target y

```text
retention_100
```

### Output

```text
outputs/trained_model.pkl
outputs/model_metrics.json
outputs/feature_importance.csv
```

### Metrics

```text
MAE
RMSE
R2
```

Vì data ít/synthetic, không yêu cầu metric đẹp. Quan trọng là pipeline chạy end-to-end.

### Acceptance test

```text
[ ] Model train không lỗi.
[ ] trained_model.pkl được tạo.
[ ] model_metrics.json được tạo.
[ ] feature_importance.csv được tạo.
[ ] Model predict được cho sample mới.
[ ] Không yêu cầu R2 cao ở MVP.
```

---

## Phase 4 — Recommendation engine

### Mục tiêu

Đây là phần quan trọng nhất: AI recommend điều kiện chế tạo tiếp theo.

### File cần code

```text
src/recommend.py
```

### Input

```text
trained_model.pkl
master_dataset.csv
search_space config
```

### Search space MVP

```yaml
si_content:
  min: 55
  max: 75
  step: 5

mxene_content:
  min: 10
  max: 30
  step: 5

alginate_content:
  min: 5
  max: 15
  step: 5

drying_temp:
  min: 70
  max: 100
  step: 10

mixing_time:
  min: 30
  max: 60
  step: 15
```

### Logic recommend

```text
1. Generate nhiều candidate recipe trong search space.
2. Loại recipe trùng với sample đã có.
3. Predict retention_100 cho từng candidate.
4. Rank theo predicted_retention.
5. Xuất top 3 recipe.
```

### Lưu ý quan trọng

Ở thời điểm recommend, sample mới chưa có SEM/EDX thật. Vì vậy MVP có thể dùng một trong hai cách:

```text
Cách 1: process-only recommendation
Model chỉ dùng thông số chế tạo để recommend.

Cách 2: process + estimated SEM/EDX
Dùng giá trị trung bình hoặc model phụ để ước lượng SEM/EDX.
```

MVP nên dùng **Cách 1** cho đơn giản.

Cách giải thích trong slide:

> Ở vòng đầu, hệ thống đề xuất dựa trên điều kiện chế tạo và dữ liệu đã có. Sau khi mẫu mới được chế tạo, dữ liệu SEM/EDX và điện hóa thật sẽ được đưa ngược lại để cập nhật model.

### Output

```text
outputs/recommendations.csv
```

Ví dụ:

```csv
rank,si_content,mxene_content,alginate_content,drying_temp,mixing_time,predicted_retention,confidence,reason
1,65,20,10,80,45,82.5,medium,"Balanced Si/MXene ratio and low predicted degradation"
2,60,25,10,80,30,80.8,medium,"Higher MXene may improve conductive network"
3,70,15,10,90,45,78.9,low,"High Si content may improve capacity but has higher risk"
```

### Confidence/risk đơn giản cho MVP

Không cần uncertainty thật ngay. Có thể định nghĩa tạm:

```text
high confidence = candidate nằm gần vùng dữ liệu đã có
medium confidence = candidate nằm hơi xa vùng dữ liệu đã có
low confidence = candidate xa vùng dữ liệu đã có
```

Có thể tính bằng khoảng cách đến nearest existing sample trong feature space.

### Acceptance test

```text
[ ] recommend.py chạy độc lập được.
[ ] Sinh ra đúng top 3 recommendations.
[ ] Không recommend recipe trùng sample đã có.
[ ] Recipe nằm trong search space hợp lệ.
[ ] Có predicted_retention.
[ ] Có confidence hoặc risk level.
[ ] Có reason ngắn, dễ hiểu.
```

---

## Phase 5 — SEM feature extraction prototype

### Mục tiêu

Có demo ảnh SEM → feature số.

Không cần segmentation quá phức tạp trong MVP.

### File cần code

```text
src/sem_features.py
```

### Input

```text
data/raw/sem_images/
```

Tên ảnh nên theo sample_id:

```text
S001.png
S002.png
S003.png
```

### Xử lý đơn giản

Dùng OpenCV hoặc scikit-image:

```text
1. Read image
2. Convert grayscale
3. Denoise
4. Threshold
5. Estimate texture/particle/porosity features
6. Save sem_features.csv
```

### Feature tối thiểu

```text
particle_count
mean_particle_area
texture_score
porosity_score
edge_density
```

### Output

```text
data/processed/sem_features_extracted.csv
```

### Acceptance test

```text
[ ] Đọc được folder ảnh.
[ ] Mỗi ảnh sinh ra một dòng feature.
[ ] Có sample_id lấy từ tên file.
[ ] Không crash nếu ảnh lỗi.
[ ] Có ít nhất 3 feature số.
```

Phase này là optional cho MVP, nhưng có ảnh demo sẽ rất tốt khi trình bày.

---

## Phase 6 — Dashboard demo

### Mục tiêu

Có giao diện để chụp screenshot đưa vào slide.

### File cần code

```text
app/streamlit_app.py
```

### Dashboard gồm 4 tab

```text
Tab 1: Dataset overview
Tab 2: Model performance
Tab 3: Feature importance
Tab 4: AI recommendation
```

### Tab 1 — Dataset overview

Hiển thị:

```text
- số sample
- bảng master_dataset
- các nhóm dữ liệu: process, SEM, EDX, electrochemistry
```

### Tab 2 — Model performance

Hiển thị:

```text
- MAE
- RMSE
- R2
- predicted vs actual chart nếu có
```

### Tab 3 — Feature importance

Hiển thị top features ảnh hưởng đến `retention_100`.

Ví dụ:

```text
mxene_content
crack_density
porosity_score
si_mxene_ratio
rct
```

### Tab 4 — AI recommendation

Hiển thị top 3 recipe:

```text
Recommended recipe #1
- Si content
- MXene content
- Alginate content
- Drying temperature
- Mixing time
- Predicted retention
- Confidence
- Reason
```

Có nút UI giả lập:

```text
Accept / Modify / Reject
```

Nút chưa cần backend thật.

### Acceptance test

```text
[ ] streamlit run app/streamlit_app.py chạy được.
[ ] Hiển thị master_dataset.
[ ] Hiển thị model metrics.
[ ] Hiển thị feature importance.
[ ] Hiển thị top 3 recommendations.
[ ] Có screenshot dùng được cho slide.
```

---

## Phase 7 — End-to-end pipeline command

### Mục tiêu

Một lệnh chạy toàn bộ pipeline.

### File cần code

```text
run_pipeline.py
```

### Flow

```text
1. build_dataset
2. train_model
3. recommend
```

### Command

```bash
python run_pipeline.py
```

### Expected terminal output

```text
[1/3] Building master dataset...
Saved: data/processed/master_dataset.csv

[2/3] Training model...
Saved: outputs/trained_model.pkl
Saved: outputs/model_metrics.json
Saved: outputs/feature_importance.csv

[3/3] Generating recommendations...
Saved: outputs/recommendations.csv

Pipeline completed successfully.
```

### Acceptance test

```text
[ ] Xóa outputs rồi chạy lại vẫn sinh đủ file.
[ ] Không cần sửa path thủ công.
[ ] Chạy từ root project không lỗi.
[ ] recommendations.csv có đúng 3 dòng top recommendation.
```

---

## Phase 8 — Testing

### Mục tiêu

Đảm bảo code không chỉ chạy may mắn một lần.

### `test_dataset.py`

```text
[ ] master_dataset có sample_id.
[ ] master_dataset có target retention_100.
[ ] Không có NaN ở cột quan trọng.
```

### `test_model.py`

```text
[ ] model file được tạo.
[ ] model có thể predict.
[ ] metrics json có MAE/RMSE/R2.
```

### `test_recommend.py`

```text
[ ] recommendations.csv tồn tại.
[ ] Có đúng top 3 recipes.
[ ] predicted_retention là số.
[ ] recipe không trùng dữ liệu cũ.
```

### Acceptance test

```bash
pytest tests/
```

Pass tối thiểu:

```text
[ ] 80–100% tests pass.
```

---

## Phase 9 — Export material cho presentation

### Mục tiêu

Tạo hình ảnh/kết quả để đưa vào slide.

### Output cần có

```text
outputs/recommendations.csv
outputs/feature_importance.csv
outputs/model_metrics.json
screenshots/dashboard_overview.png
screenshots/recommendation_tab.png
```

### Slide demo nên có

```text
1. Data schema
2. Pipeline architecture
3. Dashboard screenshot
4. Top 3 recommendation example
5. Explanation: AI gợi ý điều kiện chế tạo tiếp theo
```

### Acceptance test

```text
[ ] Có ít nhất 2 screenshot dashboard.
[ ] Có bảng top 3 recommendations.
[ ] Có thể giải thích demo trong 1 phút.
```

---

# 4. Definition of Done

Code MVP được coi là đạt khi:

```text
[ ] python run_pipeline.py chạy thành công.
[ ] Có master_dataset.csv.
[ ] Có trained_model.pkl.
[ ] Có model_metrics.json.
[ ] Có feature_importance.csv.
[ ] Có recommendations.csv với top 3 recipe.
[ ] Dashboard Streamlit mở được.
[ ] Có screenshot dashboard để đưa vào slide.
```

---

# 5. Prompt giao thẳng cho coding agent

```text
Build an MVP prototype for a multimodal AI Co-Scientist system for optimizing nano-silicon/few-layer Ti3C2Tx MXene composite anode fabrication using sodium alginate binder for lithium-ion batteries.

Goal:
Create a runnable demo that takes synthetic/sample lab data, builds a master dataset, trains a regression model to predict cycling stability/capacity retention, and recommends the top 3 next fabrication conditions. The demo will be used for a first proposal presentation, so keep it simple, explainable, and runnable.

Required flow:
Input CSV data
→ build master dataset
→ train model
→ generate top 3 recommended fabrication recipes
→ show results in Streamlit dashboard

Required input CSV files:
1. process_data.csv:
sample_id, si_content, mxene_content, alginate_content, carbon_content, mixing_time, drying_temp, pressing_pressure

2. sem_features.csv:
sample_id, particle_size_mean, porosity_score, agglomeration_index, crack_density, surface_uniformity

3. edx_data.csv:
sample_id, si_percent, ti_percent, c_percent, o_percent, impurity_percent

4. electrochem_data.csv:
sample_id, initial_capacity, capacity_50, capacity_100, retention_100, coulombic_efficiency, rct

Implement:
1. src/build_dataset.py
- Load all CSVs
- Merge by sample_id
- Create derived features: si_mxene_ratio, si_ti_ratio, capacity_fade
- Save data/processed/master_dataset.csv

2. src/train_model.py
- Train RandomForestRegressor to predict retention_100
- Use process + SEM + EDX features as input
- Save trained model to outputs/trained_model.pkl
- Save metrics to outputs/model_metrics.json
- Save feature importance to outputs/feature_importance.csv

3. src/recommend.py
- Define search space for si_content, mxene_content, alginate_content, drying_temp, mixing_time
- Generate candidate fabrication recipes
- Predict retention_100 for each candidate
- Return top 3 recipes
- Avoid recommending recipes already present in the dataset
- Save outputs/recommendations.csv with rank, recipe parameters, predicted_retention, confidence, and reason

4. app/streamlit_app.py
- Show dataset overview
- Show model metrics
- Show feature importance
- Show top 3 AI recommendations
- Include Accept / Modify / Reject buttons as placeholders

5. run_pipeline.py
- Run build_dataset, train_model, and recommend end-to-end

Acceptance tests:
- python run_pipeline.py runs without error
- data/processed/master_dataset.csv is created
- outputs/trained_model.pkl is created
- outputs/model_metrics.json is created
- outputs/feature_importance.csv is created
- outputs/recommendations.csv is created
- recommendations.csv contains exactly top 3 valid recipes
- streamlit app runs and displays the recommendation table

Do not implement heavy LLM, SAM2, database, or full Bayesian Optimization in this MVP. Use synthetic data if real data is not available. Keep the code clean, readable, and presentation-ready.
```

---

# 6. Paper research map cho agent

Agent cần đọc paper theo block. Mục tiêu không phải đọc hết, mà lấy bằng chứng để giải thích vì sao từng block trong pipeline khả thi.

---

## Block A — Vật liệu Si/MXene và sodium alginate binder

### A1. Si/Ti₃C₂Tₓ MXene anode review

**Paper:** Recent progress in Si/Ti₃C₂Tₓ MXene anode materials for lithium-ion batteries  
**Link:** https://www.sciencedirect.com/science/article/pii/S2589004224024428

Agent cần extract:

```text
- Vì sao Si có capacity cao nhưng bị volume expansion.
- Vì sao MXene giúp cải thiện conductivity, ion transport, structural stability.
- Các synthesis route phổ biến của Si/MXene anode.
- Các metric electrochemical thường dùng: capacity, retention, rate capability.
```

### A2. Sodium alginate binder cho silicon anode

**Paper:** Application and Development of Silicon Anode Binders for Lithium-Ion Batteries  
**Link:** https://www.mdpi.com/1996-1944/16/12/4266

Agent cần extract:

```text
- Vai trò của binder trong silicon anode.
- Sodium alginate có nhóm carboxyl/hydroxyl giúp bonding tốt với Si.
- Binder giúp giảm pulverization, cải thiện cycling stability.
- So sánh water-based binder với PVDF/NMP nếu có.
```

### A3. Alginate-derived binder cho silicon-based anodes

**Paper:** Rational design of alginate-derived network binder for high-performance silicon-based anodes in lithium-ion batteries  
**Link:** https://www.sciencedirect.com/science/article/abs/pii/S0378775324016975

Agent cần extract:

```text
- Alginate-based network binder giúp xử lý volume expansion.
- Cơ chế cải thiện stability của silicon-based anodes.
- Các biến binder có thể trở thành input cho model.
```

### A4. Si/MXene composite example

**Paper:** Carbon Additive-Free Crumpled Ti₃C₂Tₓ MXene Framework Encapsulated Silicon Anodes  
**Link:** https://pubs.acs.org/doi/10.1021/acsaem.1c01736

Agent cần extract:

```text
- MXene framework quanh silicon giúp duy trì conductive network.
- Các thông số fabrication đáng đưa vào schema.
- Các output electrochemical để dùng làm target.
```

---

## Block B — LLM literature mining

### B1. Structured information extraction from scientific text with LLMs

**Paper:** Structured information extraction from scientific text with large language models  
**Link:** https://www.nature.com/articles/s41467-024-45563-x

Agent cần extract:

```text
- LLM có thể trích xuất structured records từ scientific text.
- Những field có thể extract trong proposal: material, synthesis condition, morphology, capacity, retention.
- Không cần implement thật ở MVP, chỉ để làm future work hoặc data curation module.
```

### B2. Solid-state synthesis extraction dataset

**Paper/dataset:** Text-mined dataset of solid-state syntheses with impurity phases using Large Language Model  
**Repo/search keyword:** solid-state-recipes-with-impurity

Agent cần extract:

```text
- LLM mining có thể tạo dataset synthesis quy mô lớn.
- Dùng làm bằng chứng rằng literature prior là khả thi.
- Cẩn thận: dataset này không phải riêng Si/MXene; chỉ dùng để justify phương pháp extraction.
```

---

## Block C — SEM / image feature extraction

### C1. Choi et al. 2025 — paper gần nhất với pipeline của ta

**Paper:** Image-Guided Microstructure Optimization using Diffusion Models: Validated with Li-Mn-rich Cathode Precursors  
**arXiv:** https://arxiv.org/abs/2505.07906

Agent cần extract:

```text
- Đây là công trình gần nhất với pipeline image-guided closed-loop optimization.
- Paper dùng SEM-derived morphology features: texture, sphericity, D50.
- Paper dùng diffusion model + particle swarm optimization.
- Mục tiêu của họ là morphology matching / target SEM morphology.
- Điểm khác của ta: ta tối ưu electrochemical performance, không chỉ morphology.
```

Câu so sánh nên dùng:

```text
Choi et al. tối ưu điều kiện tổng hợp để đạt hình thái SEM mong muốn, còn pipeline của chúng ta dùng SEM/EDX như dữ liệu trung gian để hướng đến mục tiêu cuối cùng là hiệu suất điện hóa như capacity retention và cycling stability.
```

### C2. SEM / microscopy segmentation cho battery microstructure

**Paper:** Deep learning-based segmentation of lithium-ion battery microstructures enhanced by artificially generated electrodes  
**Link:** https://www.nature.com/articles/s41467-021-26289-8

Agent cần extract:

```text
- Battery microstructure có thể segmentation bằng deep learning.
- Feature từ microstructure có thể đưa vào model.
- MVP chưa cần implement deep learning; có thể dùng OpenCV/ImageJ trước.
```

### C3. Super-resolving microscopy images of Li-ion electrodes

**Paper:** Super-resolving microscopy images of Li-ion electrodes for fine-feature quantification using GANs  
**Link:** https://www.nature.com/articles/s41524-022-00707-7

Agent cần extract:

```text
- Microscopy image có thể được xử lý để định lượng feature nhỏ như crack/fine features.
- Chứng minh hướng SEM/image feature extraction có cơ sở.
```

---

## Block D — EDX / SEM–EDS composition feedback

### D1. Quantitative SEM–EDS process for LIB electrodes

**Paper:** A comprehensive and quantitative SEM–EDS analytical process applied to lithium-ion battery electrodes  
**Link:** https://www.nature.com/articles/s41598-025-89362-w

Agent cần extract:

```text
- SEM–EDS có thể phân tích morphology + elemental distribution trong battery electrodes.
- Paper có clustering/similarity analysis cho elemental distribution.
- Dùng để justify EDX feature trong pipeline.
- EDX feature trong code: si_percent, ti_percent, c_percent, o_percent, impurity_percent, si_ti_ratio.
```

Lưu ý:

```text
EDX không dùng để đo lithium đáng tin cậy. Với đề tài này EDX dùng cho Si/Ti/C/O/impurity và distribution, không claim đo Li chính xác.
```

---

## Block E — Electrochemical feedback

### E1. General electrochemical characterization for Si/MXene/binder anodes

Agent cần lấy từ các paper Block A các metric sau:

```text
- initial_capacity
- capacity_50
- capacity_100
- retention_100
- Coulombic efficiency
- rate capability
- impedance / Rct
```

Mục tiêu code:

```text
Dùng retention_100 làm target chính.
Nếu chưa có dữ liệu 100 cycles thì dùng retention_50 hoặc capacity_fade_rate.
```

### E2. Battery characterization review hoặc paper thực nghiệm liên quan

**Paper:** Application and Development of Silicon Anode Binders for Lithium-Ion Batteries  
**Link:** https://www.mdpi.com/1996-1944/16/12/4266

Agent cần extract:

```text
- Các metric electrochemical thường dùng cho silicon anode.
- Vì sao cycling stability là target hợp lý hơn initial capacity.
```

---

## Block F — Closed-loop recommendation / active learning / Bayesian optimization

### F1. CAMEO closed-loop materials discovery

**Paper:** On-the-fly closed-loop materials discovery via Bayesian active learning  
**Link:** https://www.nature.com/articles/s41467-020-19597-w

Agent cần extract:

```text
- Closed-loop system dùng active learning để chọn thí nghiệm tiếp theo.
- Có human-machine interaction / human-in-the-loop.
- Dùng để justify recommendation engine.
```

### F2. Electrode manufacturing optimization by Duquesnoy/Franco

**Paper:** Toward High-Performance Energy and Power Battery Cells with Machine Learning-based Optimization of Electrode Manufacturing  
**arXiv:** https://arxiv.org/abs/2307.05521  
**Journal link:** https://www.sciencedirect.com/science/article/pii/S0378775323010509

Agent cần extract:

```text
- ML-assisted pipeline có thể inverse design process parameters cho electrode manufacturing.
- Đây là nhóm công trình lớn, không nên claim ta là first.
- Điểm khác của ta: low-data, experiment-facing workflow, vật liệu Si/MXene composite, SEM+EDX+electrochemical feedback.
```

### F3. ML-assisted multi-objective battery manufacturing optimization

**Paper:** Machine Learning-Assisted Multi-Objective Optimization of Battery Manufacturing from Synthetic Data Generated by Physics-Based Simulations  
**arXiv:** https://arxiv.org/abs/2205.01621

Agent cần extract:

```text
- Dùng synthetic data từ physics-based simulations để train ML model.
- Có inverse design manufacturing parameters.
- Dùng để justify việc MVP dùng synthetic data là hợp lý cho demo ban đầu.
```

---

# 7. Related work delta cần đưa vào slide

## So với Choi et al.

| Tiêu chí | Choi et al. | Đề tài của ta |
|---|---|---|
| Vật liệu | Li- and Mn-rich cathode precursors | Nano-silicon/few-layer Ti₃C₂Tₓ MXene anode |
| Dữ liệu chính | SEM morphology | SEM + EDX + electrochemistry + literature |
| Mục tiêu | Match hình thái SEM mục tiêu | Tối ưu cycling stability / capacity retention |
| Optimizer | Diffusion model + PSO | MVP: Random Forest recommender; later: GP + BO |
| Output | Điều kiện tổng hợp cho morphology target | Điều kiện chế tạo cho performance tốt hơn |

Câu chốt:

> Choi et al. là image-guided morphology optimization; đề tài này là performance-guided electrode fabrication optimization sử dụng dữ liệu đa phương thức.

---

## So với Franco/LRCS

| Tiêu chí | Franco/LRCS | Đề tài của ta |
|---|---|---|
| Phạm vi | Electrode manufacturing optimization quy mô rộng | Workflow nhỏ hơn cho vật liệu điện cực composite trong lab |
| Dữ liệu | Simulation + manufacturing parameters + electrochemical model | Literature + SEM + EDX + electrochemical experiment |
| Mục tiêu | Inverse design manufacturing parameters cho energy/power cells | Gợi ý thí nghiệm tiếp theo cho Si/MXene anode trong điều kiện dữ liệu ít |
| Định vị | Manufacturing-scale optimization | Low-data, experiment-facing AI Co-scientist |

Câu chốt:

> Đề tài không claim thay thế các framework manufacturing optimization lớn, mà tập trung vào workflow thực nghiệm nhỏ, dữ liệu ít, dành cho phát triển vật liệu điện cực composite mới.

---

# 8. Research task list cho agent

Agent cần tạo một file `literature_notes.md` với format sau cho mỗi paper:

```text
## Paper title

Link:
Year:
Material/system:
Main method:
Data used:
Target/output:
What we can reuse:
How it supports our pipeline:
Difference from our project:
```

Agent cần ưu tiên đọc theo thứ tự:

```text
P0 — Bắt buộc đọc trước:
1. Choi et al. 2025 image-guided microstructure optimization
2. Duquesnoy/Franco 2023/2024 electrode manufacturing optimization
3. CAMEO Bayesian active learning
4. Si/Ti3C2Tx MXene anode review
5. Sodium alginate binder review / silicon binder paper

P1 — Đọc để bổ sung:
6. Kato et al. 2025 quantitative SEM–EDS for LIB electrodes
7. Dagdelen et al. 2024 LLM scientific extraction
8. SEM/microstructure segmentation papers

P2 — Để sau:
9. SAM/SAM2 microscopy segmentation
10. Gaussian Process / BoTorch implementation docs
11. More specific Si/MXene experimental papers
```

---

# 9. Notes cho agent khi code

## 9.1 Không over-engineer

MVP chỉ cần chứng minh logic:

```text
data → model → recommendation → dashboard
```

Không cần build research-grade system ngay.

## 9.2 Không claim model chính xác

Vì dùng synthetic data, trong README phải ghi rõ:

```text
This MVP uses synthetic/sample data to demonstrate pipeline logic. Model performance is not scientifically meaningful until real experimental data are provided.
```

## 9.3 Recommendation trong MVP là demonstration

Ở MVP, recommendation chỉ là demo top-k candidate ranking. Bản research thật sẽ thay bằng:

```text
- Gaussian Process surrogate
- Bayesian Optimization
- uncertainty-aware acquisition function
- human-in-the-loop experiment selection
```

## 9.4 SEM/EDX mới là feature, performance mới là target

Không để agent viết sai rằng project tối ưu ảnh SEM. Đúng phải là:

```text
SEM/EDX are input features.
Electrochemical performance is the optimization target.
```

---

# 10. Minimal requirements.txt

```text
pandas
numpy
scikit-learn
joblib
streamlit
matplotlib
plotly
pytest
opencv-python
scikit-image
pyyaml
```

Nếu agent dùng XGBoost thì thêm:

```text
xgboost
```

Nếu chưa cần SEM image extraction thì có thể bỏ:

```text
opencv-python
scikit-image
```

---

# 11. README tối thiểu cần có

Agent cần viết README gồm:

```text
1. Project goal
2. Pipeline overview
3. Folder structure
4. How to install
5. How to run pipeline
6. How to run dashboard
7. What each output file means
8. Limitations of MVP
9. Next steps
```

Command mẫu:

```bash
pip install -r requirements.txt
python run_pipeline.py
streamlit run app/streamlit_app.py
pytest tests/
```

---

# 12. Slide phrase để đưa vào presentation

Có thể dùng câu này trên slide demo:

> The MVP demonstrates the core logic of the proposed AI Co-Scientist: fabrication parameters, SEM-derived features, EDX composition data, and electrochemical results are integrated into a master dataset. A regression model learns the relationship between fabrication conditions and cycling stability, then recommends the top candidate recipes for the next experimental round. The final decision remains human-in-the-loop.

Bản tiếng Việt:

> MVP mô phỏng logic cốt lõi của AI Co-scientist: thông số chế tạo, đặc trưng SEM, dữ liệu EDX và kết quả điện hóa được tích hợp thành một master dataset. Mô hình học mối liên hệ giữa điều kiện chế tạo và độ bền chu kỳ, sau đó gợi ý các điều kiện chế tạo tiềm năng cho vòng thí nghiệm tiếp theo. Quyết định cuối cùng vẫn thuộc về nhà nghiên cứu.
