# Chest X-ray Pneumonia & COVID-19 Screening — educational project

[![Improved notebook on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baconkak/Chest-X-Ray-Pneumonia-COVID-19-Screening/blob/main/Chest_Xray_Improved_Colab.ipynb)

**รุ่นปรับปรุง 23 ตุลาคม 2026:** เริ่มที่ [Chest_Xray_Improved_Colab.ipynb](Chest_Xray_Improved_Colab.ipynb)
สำหรับ **ResNet50 pretrained + ข้อมูล NIH ครบ + cache ภาพ + การทดลองน้ำหนัก loss**
คู่มือด้านล่างหัวข้อ “รุ่นปรับปรุงสำหรับ L4” อธิบายค่าตั้ง เหตุผล และตำแหน่งผลลัพธ์
Notebook เดิมด้านล่างเก็บไว้สำหรับ baseline ตามบทเรียนและการทดลอง quick

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baconkak/Chest-X-Ray-Pneumonia-COVID-19-Screening/blob/main/Chest_Xray_Pneumonia_Colab.ipynb)

โปรเจกต์ Deep Learning ปี 3: เปรียบเทียบ **MLP, Small CNN และ ResNet18 transfer learning**
บน **NIH ChestX-ray14** ตาม dataset ที่อาจารย์ระบุ

**กลับค่าเริ่มต้นเดิมแล้ว:** `PROFILE="quick"` ดาวน์โหลด NIH ชุดเต็มประมาณ **42 GB**
แล้วเลือกผู้ป่วยไม่เกิน 3,000 คนสำหรับเริ่มทดลองเหมือนเดิม
หากต้องการฝึกข้อมูลครบทั้งหมดให้ใช้ `PROFILE="full"`
ยังตรวจพื้นที่ว่างอย่างน้อย **110 GiB** เพื่อเผื่อ archive และไฟล์ที่แตกพร้อมกัน

**ขอบเขตที่ทำได้:** ทำนายการมี label Pneumonia เทียบกับการไม่มี label นี้
NIH ChestX-ray14 ไม่มี COVID-19 label จึงยังไม่รองรับการฝึกหรือวัดผล COVID-19
ชื่อ repository คงตามหัวข้อโครงงาน แต่ไม่อ้างว่าโมเดลนี้ตรวจ COVID-19 ได้
กลุ่มไม่มี Pneumonia อาจมีโรคอื่น ไม่เท่ากับ “healthy”

ดู [คู่มือใช้งานแบบละเอียด](#คู่มือใช้งาน) ด้านล่าง
และ [รายงาน Word](report_th.docx) สำหรับรายงานวิธีการ เหตุผล สูตร และการเชื่อมกับบทเรียน 6 บท
มี [ฉบับข้อความต้นฉบับ](report_th.txt) สำหรับค้นหาและสร้างเอกสารใหม่

- [Notebook สำหรับ Google Colab](Chest_Xray_Pneumonia_Colab.ipynb)
- `src/`: data audit, patient split, models, training/resume, metrics, plots, Grad-CAM
- `tests/`: ทดสอบด้วยข้อมูลสังเคราะห์ ไม่ใช่ผลประเมินทางการแพทย์
- `requirements.txt`: dependency ranges; notebook เก็บรุ่นจริงใน environment_freeze.txt

ผล quick ที่ผู้ใช้ส่งมาผ่านการตรวจแล้ว แต่ยังไม่มี final test และยังไม่ได้ฝึกรุ่นปรับปรุงนี้
คะแนนของรุ่นปรับปรุงจะเกิดเมื่อรัน notebook; ไม่รับประกันว่าเพิ่มขนาดโมเดลแล้วคะแนนจะสูงขึ้น
ใช้เพื่อการศึกษาเท่านั้น

Data provider: **NIH Clinical Center**.
[NIH download](https://nihcc.app.box.com/v/ChestXray-NIHCC) ·
[Kaggle NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data) ·
[Wang et al., CVPR 2017, ChestX-ray8 (ChestX-ray14 in appendix)](https://arxiv.org/abs/1705.02315)

## คู่มือใช้งาน

### 1. สิ่งที่ได้รับและสิ่งที่ต้องทำต่อ

โปรเจกต์นี้เตรียม pipeline สำหรับฝึกจริงบน Google Colab แล้ว
มี MLP ตามเนื้อหา feedforward, CNN ขนาดเล็ก และ ResNet18 แบบ transfer learning
มีรายงาน Word อธิบายเหตุผลใน report_th.docx แต่ยังไม่มีคะแนนทดลองจริง
ผู้ใช้ต้องดาวน์โหลดข้อมูล เลือก GPU และรัน notebook เพื่อสร้างผลของตัวเอง
อย่านำผล synthetic tests ไปอ้างว่าเป็นประสิทธิภาพตรวจโรค

ข้อจำกัดสำคัญ: NIH ChestX-ray14 ไม่มี label COVID-19
งานที่ทำคือ Pneumonia label present / absent เท่านั้น
negative อาจมีโรคอื่น; No Finding ไม่ใช่ใบรับรองว่าปอดปกติ
ชื่อ repo ตามชื่อหัวข้อเดิม ไม่ได้หมายถึงโมเดลรองรับ COVID-19 แล้ว

### 2. โครงสร้างไฟล์

| ไฟล์ | หน้าที่ |
| --- | --- |
| [Chest_Xray_Pneumonia_Colab.ipynb](Chest_Xray_Pneumonia_Colab.ipynb) | จุดเริ่มต้น ใช้งานตามลำดับ cell |
| [report_th.docx](report_th.docx) | รายงาน Word จัดหน้า พร้อมสูตรและตารางสำหรับอ่านและแก้ไข |
| [report_th.txt](report_th.txt) | ข้อความต้นฉบับของรายงาน |
| [README.md](README.md) | หน้าแรก GitHub และคู่มือใช้งานฉบับนี้ |
| [requirements.txt](requirements.txt) | dependencies สำหรับเครื่องส่วนตัว |
| [src/data.py](src/data.py) | ตรวจข้อมูล, label, patient split, preprocessing |
| [src/models.py](src/models.py) | MLP, SmallCNN, ResNet18 |
| [src/engine.py](src/engine.py) | ฝึก, checkpoint, resume, inference |
| [src/metrics.py](src/metrics.py) | AP, AUROC, threshold และ bootstrap |
| [src/plots.py](src/plots.py) | กราฟการฝึก, PR, ROC, calibration, confusion matrix |
| [src/explain.py](src/explain.py) | Grad-CAM สำหรับ ResNet18 |
| [tests/test_pipeline.py](tests/test_pipeline.py) | ทดสอบด้วยภาพสังเคราะห์ |
| [tools/build_notebook.py](tools/build_notebook.py) | สร้าง notebook ใหม่หลังแก้เนื้อหา notebook |
| [.gitignore](.gitignore) | กัน dataset, checkpoints, tokens และไฟล์ชั่วคราว |
| [tools/build_report.py](tools/build_report.py) | สร้างรายงาน Word จากข้อความต้นฉบับ |

### 3. เปิด notebook บน Colab

#### 3.1 เข้า GitHub repository:

[Repository ของโปรเจกต์](https://github.com/baconkak/Chest-X-Ray-Pneumonia-COVID-19-Screening)
กดปุ่ม Open in Colab ใน README หรือเปิดลิงก์:
[เปิด Notebook ใน Google Colab](https://colab.research.google.com/github/baconkak/Chest-X-Ray-Pneumonia-COVID-19-Screening/blob/main/Chest_Xray_Pneumonia_Colab.ipynb)
อีกวิธี: เข้า https://colab.research.google.com แล้ว File > Upload notebook
เลือก Chest_Xray_Pneumonia_Colab.ipynb จากเครื่อง

#### 3.2 File > Save a copy in Drive เพื่อเก็บฉบับของตัวเอง

#### 3.3 Runtime > Change runtime type > เลือก GPU ถ้ามี T4 ก็เพียงพอสำหรับเริ่มทดลอง

Colab ฟรีไม่รับประกันชนิด GPU ระยะเวลา หรือโควตา อย่าวางแผนว่าต้องได้ GPU เสมอ
#### 3.4 รัน cell ทีละส่วนจากบนลงล่าง อ่านผล audit และตัวอย่างภาพก่อนเริ่มฝึก

cell แรกจะนำ source จาก GitHub มาใช้ และพิมพ์ commit SHA ให้จดไว้ในรายงาน
หากแก้โค้ดในเครื่องต้อง push ก่อน หรือ upload src/ ฉบับที่แก้เข้า runtime
หากรันจากโฟลเดอร์ repo ที่มี src/ แล้ว notebook จะใช้โค้ดในโฟลเดอร์นั้น

### 4. เลือกที่เก็บผลและขนาดงาน

`USE_DRIVE=True` จะขออนุญาต mount Google Drive ใน Colab
ผลอยู่ที่ MyDrive/ChestXrayProject/<RUN_NAME>/ ไม่ต้องเก็บ dataset ลง Git
`USE_DRIVE=False` จะเก็บใน outputs/ ของ runtime ซึ่งหายได้เมื่อ runtime ถูกลบ

`PROFILE="quick"` เป็นค่าเริ่มต้นเดิมที่นำกลับมาใช้ ดาวน์โหลดชุดเต็มแล้วเลือกผู้ป่วย 3,000 คน
ใช้ `RUN_NAME="nih_quick_seed42_v2"` เพื่อแยกจากงาน sample หรือ checkpoint เก่า

`PROFILE="sample"` เป็นทางเลือกสำหรับดิสก์น้อย ใช้ชุด NIH Random Sample เวอร์ชัน 4 ทั้ง 5,606 ภาพ
ดาวน์โหลดประมาณ 2.25 GB และแบ่ง grouped 70/15/15 ตามผู้ป่วย โดยไม่สุ่มตัดเหลือ 3,000 คนอีก
มี positive Pneumonia เพียง 62 ภาพ จึงต้องอ่าน positive support และช่วงความไม่แน่นอนประกอบ
เหมาะสำหรับเริ่มทดลอง ไม่ใช่ official benchmark หรือหลักฐานทางคลินิก

`PROFILE="quick"`: เลือกผู้ป่วยสุ่มไม่เกิน 3,000 คนจากชุดเต็ม แบ่ง grouped 70/15/15
ใช้สำหรับทดลองให้ระบบทำงานก่อน ไม่ใช่ผล official benchmark
การจำกัดผู้ป่วยไม่ลดขนาดดาวน์โหลด Kagglehub ซึ่งยังดึง dataset เต็ม
`PROFILE="full"`: ข้อมูลครบ, official test list, แบ่ง val 20% ของ train_val ตามผู้ป่วย
เปลี่ยน RUN_NAME ด้วยเมื่อเปลี่ยน profile หรือ config

ค่าเริ่มต้น: epochs=8, head_epochs=2, batch=32, image_size=224, seed=42
เปลี่ยนได้ใน TrainConfig ก่อนเริ่มงานใหม่
ถ้า GPU memory ไม่พอ ให้ลด batch_size เป็น 16 หรือ 8 และใช้ชื่อ run ใหม่
หากต้องทดลองหลาย seed บน split เดิม ให้คง seed ใน make_splits ไว้ 42
แต่เปลี่ยน CFG.seed เป็น 43/44 หลังสร้าง split (แยก seed การแบ่งกับ seed การฝึก)
บันทึก split hash เดิมเพื่อยืนยันว่าเทียบข้อมูลชุดเดียวกัน

### 5. เตรียม dataset ตามที่อาจารย์เลือก

แหล่งหลัก:
[NIH Chest X-rays บน Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
แหล่งต้นฉบับ NIH:
[ดาวน์โหลดจาก NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC)
ผู้ให้ข้อมูลคือ NIH Clinical Center; ดูข้อกำหนด attribution ใน DATA_SOURCES.txt

วิธี A (อัตโนมัติ): `DOWNLOAD_DATA=True` ใน notebook
ค่าเริ่มต้น `PROFILE="quick"` ใช้ `nih-chest-xrays/data/versions/3` ดาวน์โหลดชุดเต็มประมาณ 42 GB
ตรวจค่า **Free disk** ก่อนเริ่ม ต้องเหลืออย่างน้อย 110 GiB แม้ดิสก์รวมจะเป็น 210 GB
`PROFILE="full"` ดาวน์โหลดชุดเดียวกัน แต่ฝึกทุกผู้ป่วยและใช้ official split
เมื่อ `PROFILE="sample"` ใช้ `kagglehub.dataset_download("nih-chest-xrays/sample/versions/4")`
ดาวน์โหลดเฉพาะชุดเล็กและบันทึก path/version ที่ได้ ต้องมีพื้นที่ว่างอย่างน้อย **10 GiB**
เพื่อเผื่อ archive, ไฟล์ที่แตก และผลการฝึก ไม่ได้หมายความว่า dataset ขนาด 10 GiB
`quick` และ `full` ยังใช้ชุดเต็มเวอร์ชัน 3 และต้องมีพื้นที่ว่างอย่างน้อย **110 GiB**
หากไม่มีพื้นที่ให้ใช้ `sample` หรือใช้ไฟล์ที่แตกไว้แล้วผ่านวิธี B
หาก Kaggle ขอ login ให้ทำตาม https://github.com/Kaggle/kagglehub#authenticate
ใช้ kagglehub.login() ใน cell ชั่วคราวหรือระบบ Secrets ที่เหมาะสม ไม่เขียน token ลงไฟล์
ถ้า download ล้มเหลว/มีข้อกำหนดให้ยอมรับ ให้เปิดหน้า dataset ตรวจบัญชีและสิทธิ์ก่อน

วิธี B (มีไฟล์แล้ว): `DOWNLOAD_DATA=False`
ตั้ง DATA_ROOT=Path("/content/nih-chestxray14") หรือ path ที่มีข้อมูลจริง
ดาวน์โหลดและแตก archive ทุกชั้นก่อนใช้งาน ไม่ให้เหลือแต่ไฟล์ zip/tar
ตัวอย่างโครงสร้าง:
```text
nih-sample/
  sample_labels.csv
  sample/
    images/
      00000013_005.png

# หรือชุดเต็ม
nih-chestxray14/
  Data_Entry_2017.csv            (หรือ Data_Entry_2017_v2020.csv แต่ต้องมีเพียงหนึ่งไฟล์)
  train_val_list.txt            (จำเป็นสำหรับ full)
  test_list.txt                 (จำเป็นสำหรับ full)
  images_001/
    images/
      00000001_000.png
  images_002/
    images/
      ...
```

ตัวอ่านค้น PNG ใน subfolder ได้ ไม่ต้องย้ายภาพมารวมกัน
ใช้ `sample_labels.csv` หรือ `Data_Entry_2017*.csv` เพียงหนึ่งไฟล์ใน DATA_ROOT
sample ต้องมีภาพครบตาม metadata ของชุด sample และไม่ใช้ official test list ของชุดเต็ม
ถ้ามี PNG ชื่อซ้ำในหลายโฟลเดอร์จะหยุด เพื่อไม่เลือกภาพผิดเงียบ ๆ
quick ยอมรับบาง archive ได้เมื่อ `ALLOW_PARTIAL=True` และรายงาน missing images
full จะไม่ยอมรับภาพขาด, official lists ขาด, lists ซ้อนกัน หรือผู้ป่วยข้ามชุด
อย่าใช้ชุด pediatric Pneumonia หรือ COVID dataset อื่นแทนโดยตั้งชื่อว่า NIH

ถ้าข้อมูลเต็มใหญ่เกิน Colab ทางเลือกคือ Kaggle Notebook แล้ว Add Input เป็น dataset เดิม
ตั้ง `DOWNLOAD_DATA=False`, `USE_DRIVE=False` และ DATA_ROOT เป็น path ใต้ /kaggle/input ที่เห็นจริง
ตั้ง OUT ให้อยู่ /kaggle/working เพื่อให้เก็บผลได้ สภาพแวดล้อมนี้ต้องเลือก accelerator ตามที่บัญชีมี
ไม่ได้รับประกันโควตาหรือ GPU ของ Kaggle; เช็กในหน้าสร้าง notebook ของบัญชีตนเอง

#### หาก Colab เดิมดิสก์เต็มไปแล้ว

หากเพิ่มดิสก์เป็น 210 GB แล้วและพื้นที่ว่างเกิน 110 GiB สามารถใช้ชุดเต็มได้โดยไม่ต้องลบ runtime
หากดาวน์โหลดและแตกไฟล์ครบแล้ว ให้ใช้ `DOWNLOAD_DATA=False` และชี้ `DATA_ROOT` ไปยังข้อมูลเดิม
ขั้นตอนล้าง runtime ด้านล่างใช้เมื่อพื้นที่ยังไม่เพียงพอและต้องการเริ่มใหม่ด้วย sample เท่านั้น

1. เก็บไฟล์ผลลัพธ์ที่ต้องการไว้ใน Drive และบันทึก notebook ก่อน
2. เลือก **Runtime → Disconnect and delete runtime** เพื่อทิ้งดิสก์ชั่วคราวที่เต็ม
   การกระทำนี้ลบไฟล์ภายใน runtime และไม่ลบไฟล์ที่เก็บไว้ใน Google Drive
3. เปิด notebook ฉบับล่าสุดจากปุ่ม Open in Colab ด้านบน แล้ว Save a copy in Drive
4. เชื่อม GPU ใหม่ รันจากต้น ใช้ `PROFILE="sample"` และ `RUN_NAME="nih_sample_seed42_v1"`
5. ตรวจให้ช่อง Dataset แสดง `nih-chest-xrays/sample/versions/4` ก่อนเริ่มดาวน์โหลด

ห้ามใช้ run/checkpoint เดิมจาก dataset เต็มมารวมกับ sample เพราะ split และข้อมูลต่างกัน
รายงาน Word อธิบาย quick/full ซึ่งตรงกับค่าเริ่มต้นที่นำกลับมาใช้ ส่วน sample เป็นตัวเลือกเพิ่มเติม

### 6. ตรวจข้อมูลก่อนฝึก

cell audit จะพิมพ์ images, patients, positive_images และ prevalence ของทุก split
แต่ละ split ต้องมีทั้งสอง class ถ้า positive น้อยมากให้เพิ่มจำนวนผู้ป่วย/ภาพ
ระบบใช้ Patient ID แยกผู้ป่วยก่อน augmentation
ภาพคนเดียวกันไม่อยู่ทั้ง train/validation/test; ภาพหลายโรคยังเก็บไว้
เช่น Pneumonia|Effusion มี target=1; Effusion อย่างเดียวมี target=0
ตรวจภาพเสียทั้งหมดด้วย Pillow verify และดูตัวอย่างภาพ augmentation
การตรวจชื่อไฟล์ไม่ใช่การตรวจ near-duplicate ทุกชนิด: ดูข้อจำกัดในรายงาน
ไม่ส่งไฟล์ Sheet ที่ใช้เรียนขึ้น GitHub รายงานอ้างชื่อบทและเลขหน้าเท่านั้น

### 7. การฝึกและการกลับมาฝึกต่อ

ครั้งแรกใช้ `RESUME=False` และ RUN_NAME ใหม่
MLP, SmallCNN, ResNet18 จะฝึกตามลำดับ เก็บ learning curves และ history
ResNet18 ต้องดาวน์โหลด ImageNet weights ครั้งแรก ต้องมีอินเทอร์เน็ต
ไม่แอบเปลี่ยนเป็น random weights เมื่อ download ไม่ได้
ค่า `pretrained=False` ใช้ได้สำหรับ ablation แต่ต้องใช้ run แยกและอธิบายว่าไม่ใช่ transfer learning

เมื่อ Colab หลุด:

1. เชื่อม GPU ใหม่ รัน setup และ mount Drive
2. ใส่ RUN_NAME เดิม, profile/config เดิม และข้อมูล version เดิม
3. ตั้ง `RESUME=True` แล้วรันตั้งแต่ต้นตามลำดับ
4. ต้องมีทั้ง best.pt และ last.pt ของโมเดลที่ resume

หากหลุดก่อนเริ่มบางโมเดล ระบบจะฝึกโมเดลนั้นจากต้นเมื่อยังไม่มี last.pt
checkpoint เก็บทุกจบ epoch; ถ้าหลุดกลาง epoch งานตั้งแต่ checkpoint ล่าสุดจะถูกทำใหม่
กลับมาฝึกต่อด้วย optimizer, scheduler, scaler และ RNG เดิม ไม่ใช่แค่ load weights
resume ไม่ยอมให้เปลี่ยน config/split; ถ้าต้องการทดลองเพิ่มให้เริ่ม RUN_NAME ใหม่
best.pt เป็นโมเดลที่เลือกจาก validation, last.pt เป็นสถานะล่าสุดสำหรับ resume
โมเดลที่ early stop แล้วจะไม่ฝึกเพิ่มเมื่อ resume

### 8. การประเมินผล

อ่าน validation_comparison.csv เลือกโมเดลด้วย validation AP เท่านั้น
threshold มาจาก validation: precision สูงสุดภายใต้ recall >= 0.90
tie ใช้ threshold สูงสุด ส่วน tie ของโมเดลใช้ลำดับ MODELS ที่ประกาศไว้
0.90 เป็นเป้าทดลอง ไม่ใช่ข้อรับรองทางการแพทย์
Notebook จะเขียน locked_selection.json ก่อนเปิด test

เมื่อปิดการปรับทุกอย่างแล้ว ตั้ง `RUN_FINAL_TEST=True` และรันส่วน final test
ค่าเริ่มต้น False เพื่อไม่เผลอใช้ test เลือกโมเดล
test_comparison.csv จะมี AP, AUROC, accuracy, precision, recall, specificity, F1,
TN/FP/FN/TP, threshold, Brier score และช่วง bootstrap ของ AP
undefined ratio จะว่าง/None ไม่บังคับให้กลายเป็นคะแนนดี
all_predictions.csv เก็บคะแนนและ ID สำหรับตรวจซ้ำ
*_errors.csv เก็บ FP/FN สำหรับอภิปรายข้อผิดพลาด
*_view_slices.csv แยก AP/PA เมื่อ metadata มี View Position
*_calibration.csv มีจำนวนภาพต่อ bin เพื่ออ่าน reliability diagram
paired_ap_delta.json เปรียบเทียบ AP โมเดลที่เลือกกับ MLP แบบ paired patient bootstrap
bootstrap 200 ครั้งเป็นค่าเริ่มต้น ถ้าต้องการช่วงที่เสถียรกว่าเพิ่มเป็น 1,000–2,000 ก่อน final run

อย่าแก้ threshold จาก test หรือเลือกโมเดลใหม่เพราะ test ดีกว่า
หาก final test ถูกบันทึกแล้ว notebook จะหยุดเมื่อสั่งประเมินซ้ำ ให้เปิด CSV เดิมอ่าน
ถ้ารันล้มเหลวก่อน CSV สรุปสำเร็จ ตรวจส่วนที่ล้มและรันให้ครบ โดยไม่เปลี่ยน model/threshold

### 9. ทดลองภาพเดี่ยวและ Grad-CAM

ตั้ง IMAGE_PATH ในส่วนท้ายเป็น path PNG/JPEG แล้วรัน
จะรายงาน pneumonia label score และ threshold ของ ResNet18
score ไม่ได้ผ่าน clinical calibration; ห้ามตีความว่าเป็นโอกาสป่วยจริงของผู้ป่วย
COVID-19 จะถูกแสดงว่า not supported
Grad-CAM เป็นเพียงคำอธิบายการตอบสนองของโมเดล ไม่ใช่ขอบเขตรอยโรค

### 10. เขียนผลรายงานและส่งงาน

อ่าน report_th.docx แล้วเติมตารางผลจากไฟล์จริง ระบุว่า quick/full และ version ใด
ใส่ชื่อผู้จัดทำ รหัสนักศึกษา รายวิชา ผู้สอน ตามข้อมูลจริง
แนบ learning curves, PR/ROC, confusion matrix, error analysis และข้อจำกัด
อธิบายด้วยว่าผลต่างมีความไม่แน่นอน ไม่สรุปว่า ResNet ดีกว่าเสมอ
หากยังไม่ได้ฝึก ให้ระบุว่ายังไม่มี experimental results ตามความจริง
เก็บ checkpoints และผลใน Drive; ข้อมูลดิบไม่ควรขึ้น Git

### 11. ใช้ในเครื่องส่วนตัว (ทางเลือก)

ติดตั้ง Python ที่รองรับ dependency และสร้าง environment แยก
ติดตั้ง torch กับ torchvision คู่ที่ตรงกันจาก https://pytorch.org/get-started/locally/
จากนั้นในโฟลเดอร์ Project:
```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

เปิด ipynb ด้วย Jupyter/VS Code ตั้ง `USE_DRIVE=False` และ DOWNLOAD_DATA/DATA_ROOT ตามข้อมูลจริง
tests ใช้ข้อมูลสังเคราะห์เพื่อเช็ก workflow ไม่ต้องมี dataset หรือโหลด pretrained weights
หากแก้ tools/build_notebook.py ให้รัน python tools/build_notebook.py เพื่อปรับ ipynb ให้ตรงกัน

### 12. ปัญหาที่พบบ่อย

- **No module named src:** รัน cell setup จากต้น และตรวจว่ามี repo/src จริง
- **torchvision import error:** ใช้คู่ torch/torchvision ที่เข้ากัน อย่าอัปเดตทีละตัว
- **CUDA out of memory:** ลด batch, รีสตาร์ต runtime, ใช้ชื่อ run ใหม่ถ้าเปลี่ยน config
- **No PNG found:** แตก archive ซ้อนให้ครบและตั้ง DATA_ROOT ถูกที่
- **Missing metadata:** ใช้ Data_Entry_2017*.csv ของ NIH ไม่ใช่ CSV dataset อื่น
- **Too few positive patients:** เพิ่มข้อมูล/ผู้ป่วย อย่าแก้ด้วยการ split แบบสุ่มรายภาพ
- **Checkpoint mismatch:** ใช้ config และ split เดิม หรือสร้างงานใหม่
- **Drive ช้า:** อ่านภาพจาก local runtime และเก็บเฉพาะ checkpoint/results บน Drive
- **NaN loss/gradient:** หยุดตรวจภาพ/learning rate ไม่แทน NaN ด้วยศูนย์
- **AMP overflow ช่วงแรก:** ระบบข้าม update และลด scale อัตโนมัติ ดู amp_skipped_steps ใน history ถ้าเกิดซ้ำมากจะหยุด ให้ตรวจข้อมูล/learning rate หรือเริ่ม run ใหม่ด้วย `amp=False`
- **Download ต้อง login:** เปิด dataset และทำตามระบบของ Kaggle ไม่เก็บ token ใน Git

อ่านเพิ่มเติม: [Colab FAQ](https://research.google.com/colaboratory/faq.html)

## รุ่นปรับปรุงสำหรับ L4

### เหตุผลที่เปลี่ยน

จากผล `nih_quick_seed42_v1` ที่ผู้ใช้ส่งมา ResNet18 มี validation AP ประมาณ 0.0896,
AUROC ประมาณ 0.6762 และเมื่อเลือก threshold ให้ recall อย่างน้อย 0.90 ได้ TP=15, FN=1,
FP=1,069, TN=594 จาก validation 1,679 ภาพ ซึ่งมี positive เพียง 16 ภาพ
จึงยังสรุปไม่ได้ว่าใช้งานคัดกรองจริงได้ และยังไม่มี final test ในชุดไฟล์นั้น
รอบ quick มี training positive เพียง 64 ภาพและ `pos_weight` ประมาณ 112.19

การปรับครั้งนี้เป็นสมมติฐานที่ต้องทดสอบ ไม่ใช่ผลรับประกันว่า accuracy/AP จะสูงขึ้น:

| เปลี่ยนอะไร | เหตุผลและวิธีตรวจ |
| --- | --- |
| เพิ่ม ResNet50 pretrained | ประมาณ 23,510,081 พารามิเตอร์หลังเปลี่ยนหัวเป็นหนึ่ง logit เทียบ ResNet18 11,177,025; เพิ่มความสามารถเรียนรู้ feature แต่เสี่ยง overfit จึงเทียบใน split เดียวกัน |
| เพิ่ม DenseNet121 เป็นตัวเลือก | 6,954,881 พารามิเตอร์หลังเปลี่ยนหัว; เชื่อม feature ข้ามชั้น ลึกกว่าแต่พารามิเตอร์น้อยกว่า ResNet18 ไม่อ้างว่าเป็นโมเดลใหญ่กว่า |
| ใช้ PROFILE="full" | ใช้ภาพและผู้ป่วยทั้งหมดที่ผ่าน audit เพิ่มจำนวน positive จริง ไม่สร้างภาพซ้ำเพื่ออ้างว่าเพิ่มข้อมูล; official test list คงไว้ และแยก validation ตามผู้ป่วย |
| LR หัว 1e-4 / fine-tune 1e-5 | ลดจาก quick เดิม 1e-3 / 1e-4 เพื่อลดการเปลี่ยนน้ำหนักรุนแรง; ใช้ scheduler ลด LR เมื่อ validation AP ไม่ดีขึ้น |
| Loss weights 10 กับ 1 | เทียบ weighted BCE ระดับปานกลางกับ BCE ปกติ แทนการยึดอัตราส่วน 112 เป็นคำตอบ; เลือกด้วย validation เท่านั้น |
| ทดลอง weights 30 และ auto ได้ | เปิด RUN_EXTRA_ABLATIONS=True ก่อนเริ่ม run ใหม่; auto คำนวณจาก train ของรอบ full จริง ไม่ใช้ 112 จาก quick ค้างไว้ |
| Cache pad+resize RGB 224 | ย่อภาพต้นฉบับครั้งเดียวลง local disk แบบ PNG ก่อน random affine; ไม่ cache augmentation ลดการถอดภาพใหญ่ซ้ำทุก epoch |
| 4 workers / pinned memory / non-blocking transfer | ให้ CPU เตรียมภาพหลายชุดล่วงหน้า; เอาการคัดลอก prediction ของ train ที่ไม่ได้ใช้และการตรวจพารามิเตอร์ทีละตัวทุก batch ออก เหลือตรวจ loss/grad ทุก batch และพารามิเตอร์ท้าย epoch |
| Batch 64 / AMP scale 1024 | ค่าเริ่มต้นสำหรับ L4 224 px; ลด overflow ตอนเริ่มเมื่อเทียบกับ scale 65,536 และบันทึกจำนวน update ที่ข้าม ไม่มีการซ่อน non-finite loss |
| 20 epochs / patience 5 | ให้ fine-tune มีโอกาสเรียนรู้มากขึ้นโดยเลือก best checkpoint จาก validation AP; อาจหยุดก่อนครบและไม่ได้เลือก epoch ล่าสุดอัตโนมัติ |
| Seeds 42/43/44 เป็นตัวเลือก | SPLIT_SEED=42 คงเดิม เปลี่ยนเฉพาะ training seed; เลือก experiment ด้วย mean AP และใช้ seed แรกที่ประกาศ ไม่เลือก seed ที่คะแนนดีที่สุด |

ResNet50 ใช้ ImageNet IMAGENET1K_V2; ResNet18 และ DenseNet121 ใช้ IMAGENET1K_V1
pretraining คนละ recipe จึงเป็นการเปรียบเทียบ pipeline ที่ใช้งานจริง ไม่ใช่การแยกผลของสถาปัตยกรรมอย่างเดียว
ภาพใช้ padding รักษาขอบเขตภาพแทน center crop ของ ImageNet เพราะไม่ต้องการตัดส่วนของทรวงอกออก
ไม่มีหลักฐานว่าการใช้ normalization จาก ImageNet เหมาะที่สุดกับ NIH; เป็นค่าคงที่ที่ใช้เริ่ม transfer learning
ฝึกหัว 2 epochs แล้วเปิด layer4 ของ ResNet หรือ denseblock4 และ norm5 ของ DenseNet
คงสถิติ BatchNorm จาก pretrained, dropout 0.3, AdamW, weight decay 1e-4 และ clip norm 5

### Loss และเกณฑ์ตัดสิน

ยังใช้ `BCEWithLogitsLoss` เพราะทำนาย label เดียวแบบ binary ด้วยหนึ่ง logit
loss รับ logit โดยตรงและรวม sigmoid อย่างเสถียรทางตัวเลข ไม่ใส่ sigmoid ก่อน loss
CrossEntropyLoss เหมาะกับการจัดคลาสแบบเลือกหนึ่งจากหลายคลาส แต่ไม่จำเป็นสำหรับหัวหนึ่ง logit นี้
`positive_weight=1` เท่ากับ BCE ไม่ถ่วงน้ำหนัก; ค่ามากขึ้นเพิ่มโทษเมื่อพลาด positive
ไม่ใช้ oversampling ร่วมด้วยเพื่อไม่เพิ่มผลของ class imbalance ซ้ำ
weighted score ต้องประเมิน calibration ไม่ตีความเป็นโอกาสป่วยทางคลินิกโดยตรง

ไฟล์ `validation_operating_points.csv` เปรียบเทียบ threshold ที่เลือกจาก validation
สำหรับ recall เป้าหมาย 0.50, 0.80, 0.90 และ threshold คงที่ 0.5 เพื่อให้เห็น false positives
เกณฑ์หลักยังเป็น precision สูงสุดภายใต้ validation recall อย่างน้อย 0.90
ไม่เปลี่ยนเกณฑ์อัตโนมัติหลังเห็น test และ test recall ไม่รับประกันว่าจะถึง 0.90
ใช้ AP เป็นเกณฑ์จัดอันดับหลักร่วมกับ precision/recall/specificity และจำนวน FP/FN
accuracy สูงมากอาจเกิดจากทายลบทุกภาพได้เพราะ positive มีน้อย

### เปิดและรัน

1. เปิด [notebook รุ่นปรับปรุงบน Colab](https://colab.research.google.com/github/baconkak/Chest-X-Ray-Pneumonia-COVID-19-Screening/blob/main/Chest_Xray_Improved_Colab.ipynb) แล้ว Save a copy in Drive
2. Runtime → Change runtime type → L4 GPU แล้วตรวจผลชื่อ GPU จาก cell แรก
3. ค่าเริ่มต้น RUN_NAME="nih_full_resnet50_v3", batch=64, workers=4, full data และ 3 trials
4. Drive ต้องเป็นบัญชีที่ต้องการเก็บผล อนุญาตการเชื่อม Drive เมื่อ Colab ถาม
5. Run all ตามลำดับ โค้ด clone ลง `/content/chest-xray-improved` และ fetch CODE_REF ป้องกันใช้ src รุ่นเก่าค้างจาก notebook เดิม
6. ถ้าข้อมูลชุดเต็มที่แตกเสร็จอยู่ใน Kagglehub cache ของ runtime นี้ จะใช้ต่อ; ถ้า runtime ใหม่ต้องดาวน์โหลดและแตกใหม่
7. audit ทุกภาพ ตรวจ patient leakage แล้วสร้าง cache ภาพเล็กใน `/content/nih_cache`; ต้นฉบับไม่ถูกลบ
8. การฝึกแสดง progress bar, img/s และ peak GPU memory แต่ละ epoch; ใช้ความเร็วจริงประมาณเวลาที่เหลือ อย่าใช้ปริมาณ GPU RAM อย่างเดียวตัดสินว่าฝึกเร็วหรือช้า
9. หลังครบ trials จะสรุป validation และล็อกตัวเลือก `RUN_FINAL_TEST=False` ตามค่าเริ่มต้น จึงยังไม่เปิด test
10. เมื่อยุติการปรับโมเดลค่อยเปิด final test เพื่อรายงานผล; sentinel ป้องกันฝึกซ้ำใน run เดิมหลังเปิด test

งาน full มีภาพมากกว่า quick อย่างชัดเจน แม้ pipeline เร็วขึ้น เวลาฝึกทั้งงานอาจเพิ่มขึ้น
แต่ละ trial สูงสุด 20 epochs; จำนวนรวมค่าเริ่มต้นสูงสุด 60 epochs ก่อน early stopping
RUN_DENSENET และ RUN_EXTRA_ABLATIONS เพิ่มเวลา; seeds 3 ตัวเพิ่มงานเป็น 3 เท่า
ค่าใหม่ไม่ใช่การจอง L4 เสมอไป ขึ้นกับทรัพยากรของ Colab ในขณะนั้น
ถ้า CUDA out of memory ให้เริ่มชื่อ run ใหม่และลด batch เป็น 32 ไม่เปลี่ยน batch แล้ว resume checkpoint ที่ config ต่างกัน

### ผลลัพธ์และการกลับมาฝึกต่อ

ผลอยู่ `MyDrive/ChestXrayProject/nih_full_resnet50_v3/`:

- `experiment_plan.json`: แผน config, seeds, loss weights และ split hash
- `data_audit.json`, `splits/`: จำนวนภาพ/positive และหลักฐานไม่มีผู้ป่วยข้ามชุด
- `validation_comparison.csv`: metrics ของทุก trial
- `validation_seed_summary.csv`: mean/std AP ข้าม seeds; std ว่างเมื่อมี seed เดียว
- `validation_operating_points.csv`: precision/recall/FP/FN หลายเกณฑ์จาก validation
- `<experiment>_seed42/<model>/`: `run.json`, `history.csv`, `learning_curves.png`, `best.pt`, `last.pt`, `validation.json`, `val_predictions.csv`
- `locked_selection.json`: trial, model และ threshold ที่เลือก
- `final_test/`: เกิดเมื่อเปิด final test พร้อม predictions, error review, plots, comparison และ paired patient bootstrap AP

ตัวอย่าง checkpoint: `resnet50_pw10_seed42/resnet50/best.pt`
checkpoint รุ่นนี้มี config ใหม่ ไม่ resume ต่อจาก `nih_quick_seed42_v1`
ถ้าจะกลับไปใช้ checkpoint เก่า ให้ใช้ code commit ที่เก็บใน `run.json` ของงานเก่านั้น
resume รุ่นใหม่ใช้ RUN_NAME และ config เดิมทุกค่า พร้อม `RESUME=True`
รองรับขอบ epoch; หากหลุดกลาง epoch จะฝึก epoch นั้นใหม่
workers ไม่ persistent เพื่อให้ state ของ generator และ RNG ที่ checkpoint ไว้ทำซ้ำ augmentation เมื่อ resume ได้
ไม่รับประกัน bitwise equality ข้าม GPU, CUDA หรือรุ่นไลบรารีต่างกัน

การเปลี่ยน quick เป็น full เปลี่ยนกลุ่มผู้ป่วยและ validation จึงไม่เปรียบเทียบ AP กับ 0.0896 ตรง ๆ
ให้เปรียบเทียบ ResNet18 baseline กับ ResNet50 ภายในรอบ full เดียวกัน
ผล bootstrap และ seeds ควรประกอบการสรุป ไม่อ้างว่าดีขึ้นอย่างมีนัยสำคัญจากคะแนนจุดเดียว
รุ่นนี้ยังเป็น Pneumonia label present/absent ไม่มี COVID-19 label และไม่ใช่ระบบวินิจฉัย

อ้างอิงวิธีและ API: [ResNet50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html),
[DenseNet121](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html),
[DataLoader](https://docs.pytorch.org/docs/stable/data.html),
[BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

## ตรวจภาพผิดและทดลองปลด backbone ทีละช่วง

ใช้ [Chest_Xray_Backbone_Diagnostics.ipynb](Chest_Xray_Backbone_Diagnostics.ipynb)
หลังรอบ `nih_full_resnet50_v3` ฝึกเสร็จ โดยรัน cell ต่อท้ายใน runtime เดิมที่ยังมีภาพและ cache
notebook นี้ไม่ดาวน์โหลด dataset อัตโนมัติ และไม่เปิด final test

1. โหลด train/validation/test manifests เดิมและตรวจ SHA256 กับ `data_audit.json`
   รวมทั้งตรวจ patient ID, label และภาพไม่ซ้ำข้ามชุด ไม่เรียกสุ่มแบ่งข้อมูลใหม่
2. อ่าน predictions ของ ResNet18 ที่เลือกไว้เพื่อสร้างรายการ FP/FN/TP/TN ของ validation
   พร้อม metrics แยกตาม view/เพศและจำนวนโรคที่พบร่วมกัน ขนาดกลุ่ม/prevalence ต้องอ่านร่วมกัน
3. สร้างภาพ original / input224 / Grad-CAM สำหรับ FN score ต่ำ, FP score สูง,
   FP สุ่ม และ TP/TN เปรียบเทียบ โดยหลีกเลี่ยงผู้ป่วยซ้ำในแต่ละกลุ่ม
   CSV มีข้อผิดพลาดทั้งหมด แต่ contact sheets เป็นตัวอย่างเพื่อสำรวจ ไม่ใช่ตัวแทนทั้งชุด
4. บันทึกข้อสังเกตใน `review_annotations.csv`: ตรวจขอบภาพ ตัวหนังสือ อุปกรณ์ คุณภาพภาพ
   และผลของการย่อ โดยไม่แก้ label เอง ไม่ใช้ CAM เพื่อยืนยันโรคหรือสาเหตุ
5. เริ่มทั้งสาม arm จาก `resnet18_pw10_seed42/resnet18/best.pt` ไฟล์เดียวกัน
   คง seed42, loss weight10, batch64, input224 และ validation เดิม

| Trial | Epochs 1–3 | Epochs 4–6 | Epochs 7–9 |
| --- | --- | --- | --- |
| control_layer4 | layer4 + head | layer4 + head | layer4 + head |
| progressive_layer34 | layer4 + head | layer3+4 + head | layer3+4 + head |
| progressive_all | layer4 + head | layer3+4 + head | ทุกชั้น |

LR ของ head=1e-5, layer4=3e-6, layer3=9e-7, layer2=2.7e-7,
layer1=8.1e-8 และ stem=2.43e-8; ชั้นก่อนหน้าเปลี่ยนน้ำหนักเบากว่า
BatchNorm running statistics คงที่ทุก arm เพื่อแยกผลของการปลดชั้น
Adam moments ถูกเก็บต่อเนื่องเมื่อเปิดชั้นเพิ่ม และ reset LR/scheduler ที่ขอบช่วงเดียวกันทุก arm
ใช้ GPU เพิ่มสูงสุด 27 epochs; เวลา full-backbone ต้องวัดจริงและอาจใช้ VRAM เพิ่มขึ้น
เป็นการสำรวจด้วยงบสั้น หากไม่ดีขึ้นยังสรุปไม่ได้ว่าปลด backbone ไม่มีประโยชน์ทุกกรณี

เลือก checkpoint ด้วย validation AP ตาม min_delta เดิม และเก็บ checkpoint เริ่มต้นไว้ด้วย
ถ้าไม่มี epoch ใหม่ดีกว่าจุดเริ่ม จะได้ selected_epoch=0 ซึ่งหมายถึงนำ best v3 เดิมกลับมาใช้
เปรียบเทียบ AP และ precision/FP ที่ validation recall>=0.90 พร้อม paired patient bootstrap
ค่า interval เป็น exploratory หลังใช้ validation เลือกโมเดล ไม่ใช่หลักฐานนัยสำคัญจาก final test
มี seed เดียว จึงยังไม่ได้วัดความแปรปรวนข้ามการฝึก

ผลใหม่อยู่ `MyDrive/ChestXrayProject/nih_backbone_diagnostics_v1/`:

- `error_review/validation_errors.csv`, `validation_slices.csv`, `co_labels.csv`
- `error_review/review_manifest.csv`, `review_annotations.csv` และภาพประกอบตรวจข้อผิดพลาด
- `plan.json`: ตารางปลดชั้น ค่าตั้ง split hash และ SHA256 checkpoint จุดเริ่ม
- `<trial>/resnet18/`: best/last checkpoints, history, learning curves และ validation predictions
- `validation_comparison.csv`, `paired_validation_predictions.csv`, `provenance.json`

หากหลุด ใช้ RESUME_DIAGNOSTICS=True กับแผน/ชื่อเดิมทุกค่า
ไม่เขียนทับรอบ v3 และไม่ใช้ผล test มาปรับโมเดล
การป้องกัน worker cleanup error: ใช้ text tqdm และ DataLoader แบบ spawn เพื่อไม่สืบทอด
สถานะ notebook/CUDA ผ่าน fork; ยังคงปิด persistent workers เพื่อ checkpoint RNG ที่ขอบ epoch
อาจมีเวลาเริ่ม worker เพิ่มขึ้น แต่ไม่เปลี่ยน label หรือแอบซ่อน exception

อ้างอิง: [PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial),
[รายงานปัญหา tqdm.auto กับ DataLoader](https://github.com/pytorch/pytorch/issues/53703)
