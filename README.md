# CDC/CCI Surgical Risk Prediction

**Dự báo sớm nguy cơ biến chứng phẫu thuật nặng (Clavien-Dindo Grade IV/V), dựa trên hồ sơ bệnh nhân trước mổ.**

Đồ án cuối kỳ môn DDM501 (MLOps) — xây dựng đầy đủ một hệ thống ML production: từ sinh dữ liệu, huấn luyện có experiment tracking, REST API serving, web UI, đến containerization và giám sát.

> Repo: https://github.com/PhamLeTanThinh/CDC_Classification

---

## 1. Bài toán

Thang đo **Clavien-Dindo (CDC)** là chuẩn quốc tế để phân loại mức độ nặng của biến chứng **sau phẫu thuật** (Grade I → V, dựa trên phương pháp điều trị cần dùng). Hạn chế: đây là công cụ đo lường **hồi cứu** — chỉ biết được mức độ biến chứng sau khi nó đã xảy ra, không giúp bệnh viện chuẩn bị trước (đặt máu, giường ICU...).

**Mục tiêu của hệ thống**: dùng hồ sơ bệnh nhân **trước mổ** (tuổi, BMI, điểm ASA, bệnh nền, tính chất mổ, chỉ số cận lâm sàng) để dự báo sớm xác suất bệnh nhân sẽ rơi vào nhóm biến chứng nặng (tương ứng CDC Grade IV/V) — một hệ thống **cảnh báo sớm**, bổ sung cho thang đo CDC chứ không thay thế.

- **User**: Phẫu thuật viên, phòng Kế hoạch tổng hợp (KHTH)
- **Input**: Age, BMI, ASA_Score (I–V), Has_Diabetes, Has_HTN, Surgery_Type (chương trình/cấp cứu), PreOp_WBC, PreOp_Albumin
- **Output**: xác suất biến chứng nặng (0–1) + giải thích yếu tố ảnh hưởng (SHAP)
- **Dữ liệu**: hiện tại là **dữ liệu giả lập (synthetic)** — 5.000 ca mổ, sinh theo công thức risk-score y khoa giả lập + nhiễu ngẫu nhiên, tỷ lệ nhãn dương ~10% (mất cân bằng có chủ đích). Đây là proof-of-concept kỹ thuật, **chưa được kiểm chứng lâm sàng**.

Tài liệu tham khảo: [Dindo et al., *Classification of Surgical Complications*, Annals of Surgery 2004](<CDC-Classification of Surgical Complications.pdf>), [Clavien-Dindo Scale Explained (GYNQI)](<Clavien-Dindo-Scale-Explained.pdf>).

## 2. Kiến trúc hệ thống

Chi tiết đầy đủ (sơ đồ Mermaid, phân tích trade-off, các pattern đã cân nhắc và lý do không dùng — Lambda/Kappa, microservices, ensemble serving) nằm ở **[`docs/architecture.md`](docs/architecture.md)**. Tóm tắt:

```
Offline (training, trigger thủ công)
  data_generation.py → SurgeryPreprocessor (impute+scale) → train.py (SMOTE + LR/RF/XGBoost)
      → MLflow tracking (SQLite) → chọn model ROC-AUC cao nhất → models/best_model.joblib

Online (serving, luôn chạy)
  FastAPI (src/api/) load best_model.joblib → /predict, /predict/explain, /predict/batch
      → Prometheus /metrics → Grafana dashboard
  web/ (static UI) mount chung process với API
```

Quyết định kiến trúc chính: **monolithic**, không tách microservices, không dùng streaming (Lambda/Kappa), không ensemble nhiều model — mỗi quyết định đều được đối chiếu với framework của môn học và giải thích lý do trong `docs/architecture.md`.

## 3. Cấu trúc thư mục

```
├── src/
│   ├── config.py              # đường dẫn, constants dùng chung
│   ├── data_generation.py     # sinh dữ liệu giả lập
│   ├── preprocessing.py       # encode + KNN impute + scale (fit train-only, không leak test)
│   ├── train.py                # SMOTE + train 3 model + MLflow tracking + chọn best model
│   ├── explain.py              # SHAP summary/waterfall plot (báo cáo offline)
│   └── api/
│       ├── schemas.py          # Pydantic request/response models
│       ├── model_service.py    # load model 1 lần, predict/explain
│       └── main.py             # FastAPI app, 6 endpoint, Prometheus metrics
├── web/                        # web UI tĩnh (HTML/CSS/JS thuần, không framework)
├── data/raw/                   # dataset gốc (mock_surgery_data.csv)
├── models/                     # model đã train (gitignored — sinh ra bằng src/train.py)
├── monitoring/
│   ├── prometheus.yml          # scrape config
│   ├── alert_rules.yml         # 4 rule cảnh báo
│   └── grafana/provisioning/   # datasource + dashboard tự động cấu hình
├── docs/
│   ├── architecture.md         # System design, data flow, trade-offs
│   └── deployment-azure.md     # Hướng dẫn deploy lên Azure VM từng bước
├── tests/                      # (đang xây dựng)
├── Dockerfile                  # image tự train model lúc build, self-contained
├── docker-compose.yml          # api + mlflow + prometheus + grafana
├── CDC_CCI_DataGeneration.ipynb # notebook gốc (tài liệu tham khảo)
├── CDC_CCI_Model.ipynb          # notebook gốc (tài liệu tham khảo)
└── requirements.txt
```

## 4. Công nghệ sử dụng

| Thành phần | Công nghệ | Vì sao |
|---|---|---|
| Xử lý dữ liệu | pandas, scikit-learn (KNNImputer, StandardScaler) | Chuẩn, đủ cho dữ liệu tabular |
| Cân bằng dữ liệu | imbalanced-learn (SMOTE) | Nhãn dương chỉ ~10% |
| Model | Logistic Regression, Random Forest, XGBoost | So sánh 3 baseline phổ biến |
| Experiment tracking | MLflow (SQLite backend) | Mã nguồn mở, tự host, miễn phí |
| Giải thích mô hình | SHAP | Global (summary plot) + local (waterfall, per-request) |
| API | FastAPI | Validate input bằng Pydantic, tự sinh Swagger/OpenAPI docs |
| Web UI | HTML/CSS/JS thuần | Nhẹ, không cần build step, mount chung process với API |
| Giám sát | Prometheus + Grafana | Đúng yêu cầu rubric, dashboard tự provision |
| Container | Docker + Docker Compose | Chuẩn hoá môi trường, 1 lệnh chạy cả stack |
| Cloud | Azure VM | Tận dụng Azure for Students, chạy y nguyên docker-compose |

## 5. Kết quả mô hình (lần train gần nhất)

| Model | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| **Logistic Regression** ⭐ (được chọn) | **0.937** | **0.840** | 0.449 | 0.585 |
| XGBoost | 0.926 | 0.550 | 0.545 | 0.547 |
| Random Forest | 0.924 | 0.610 | 0.570 | 0.589 |

Model được chọn tự động theo ROC-AUC cao nhất. Recall cao (0.84) được ưu tiên hơn Precision cho bài toán y tế — thà cảnh báo dư còn hơn bỏ sót ca nguy cơ cao.

**Top yếu tố ảnh hưởng** (theo SHAP): điểm ASA cao, Albumin máu thấp, mổ cấp cứu, tuổi cao → tăng nguy cơ. Khớp với logic y khoa thực tế.

## 6. Cài đặt & chạy local

```bash
# 1. Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Cài dependencies
pip install -r requirements.txt

# 3. (Tuỳ chọn) Sinh lại dữ liệu giả lập
python -m src.data_generation

# 4. Train model (SMOTE + 3 model + MLflow tracking)
python -m src.train

# 5. (Tuỳ chọn) Sinh biểu đồ SHAP offline
python -m src.explain

# 6. Chạy API + Web UI
uvicorn src.api.main:app --reload --port 8000
```

Mở trình duyệt:
- **http://localhost:8000/** — Web UI dự báo
- **http://localhost:8000/docs** — Swagger UI

## 7. Chạy bằng Docker

```bash
docker compose up --build -d
```

| Service | URL |
|---|---|
| API + Web UI | http://localhost:8000/ |
| MLflow UI | http://localhost:5000/ |
| Prometheus | http://localhost:9090/ |
| Grafana | http://localhost:3000/ (anonymous viewer, admin/admin để edit) |

Chi tiết deploy lên Azure: **[`docs/deployment-azure.md`](docs/deployment-azure.md)**.

## 8. API Documentation

| Endpoint | Method | Mô tả |
|---|---|---|
| `/health` | GET | Health check (dùng cho Docker healthcheck) |
| `/metrics` | GET | Prometheus metrics |
| `/api/v1/model/info` | GET | Metadata model đang phục vụ (tên, ROC-AUC, thời gian train, MLflow run_id) |
| `/api/v1/predict` | POST | Dự báo 1 bệnh nhân |
| `/api/v1/predict/batch` | POST | Dự báo nhiều bệnh nhân cùng lúc |
| `/api/v1/predict/explain` | POST | Dự báo + giải thích SHAP theo từng yếu tố |

Schema đầy đủ + thử trực tiếp: xem Swagger UI tại `/docs`.

## 9. Giám sát & Alerting

- **Prometheus** thu thập: request rate, latency (p95), phân bố prediction theo risk flag, error rate — qua endpoint `/metrics`.
- **Grafana** có sẵn dashboard "CDC/CCI API Overview" (tự động provision, không cần setup tay).
- **4 alert rule** (`monitoring/alert_rules.yml`): API down, latency cao, error rate cao, và 1 rule đặc thù domain — tỷ lệ dự báo nguy cơ cao bất thường (>50%, baseline ~10%) như tín hiệu cảnh báo data drift.
- *Giới hạn đã biết*: alert chỉ hiển thị trên Prometheus UI, chưa nối Alertmanager để gửi email/Slack.

## 10. Responsible AI & giới hạn

- Dữ liệu huấn luyện là **synthetic**, chưa kiểm chứng trên dữ liệu lâm sàng thật.
- Model có giải thích được (SHAP) ở cả cấp độ toàn cục và từng ca cụ thể — hỗ trợ bác sĩ hiểu vì sao model đưa ra cảnh báo, không phải hộp đen.
- API hiện **chưa có authentication** — chấp nhận được cho demo course, không phù hợp để xử lý dữ liệu bệnh nhân thật.
- Chưa có phân tích fairness/bias theo nhóm (tuổi, loại phẫu thuật...) — ghi nhận là việc cần làm thêm.
- Chi tiết đầy đủ: `docs/architecture.md` §8.

## 11. Trạng thái dự án

| Hạng mục (theo rubric DDM501) | Trạng thái |
|---|---|
| A. Problem Definition & Requirements | ✅ Xong |
| B. System Design & Architecture | ✅ Xong (`docs/architecture.md`) |
| C. Implementation — ML Pipeline | ✅ Xong (data, preprocessing, training, MLflow) |
| C. Implementation — Deployment | ✅ Xong (API, Docker, Docker Compose) |
| C. Implementation — Monitoring | ✅ Xong (Prometheus, Grafana, alert rules) |
| D. Testing & CI/CD | 🔲 Chưa làm — bước tiếp theo |
| E. Responsible AI | 🟡 Một phần (SHAP có; fairness/privacy/ethics viết đầy đủ chưa) |
| F. Documentation | 🟡 Đang hoàn thiện (README, architecture, deployment guide đã có; API docs tự sinh qua Swagger) |

## 12. Nhóm thực hiện

*(điền thông tin thành viên nhóm ở đây)*

---

## Tài liệu tham khảo

1. Dindo D, Demartines N, Clavien P-A. *Classification of Surgical Complications: A New Proposal with Evaluation in a Cohort of 6336 Patients and Results of a Survey.* Annals of Surgery. 2004;240(2):205-213.
2. Zhou GX, Murji A, Shirreff L. *Clavien Dindo Scale of Surgical Complications.* GYNQI, 2022.
