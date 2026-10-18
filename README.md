# Chest X-ray Pneumonia & COVID-19 Screening — educational project

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baconkak/Chest-X-Ray-Pneumonia-COVID-19-Screening/blob/main/Chest_Xray_Pneumonia_Colab.ipynb)

โปรเจกต์ Deep Learning ปี 3: เปรียบเทียบ **MLP, Small CNN และ ResNet18 transfer learning**
บน **NIH ChestX-ray14** ตาม dataset ที่อาจารย์ระบุ

**ขอบเขตที่ทำได้:** ทำนายการมี label Pneumonia เทียบกับการไม่มี label นี้
NIH ChestX-ray14 ไม่มี COVID-19 label จึงยังไม่รองรับการฝึกหรือวัดผล COVID-19
ชื่อ repository คงตามหัวข้อโครงงาน แต่ไม่อ้างว่าโมเดลนี้ตรวจ COVID-19 ได้
กลุ่มไม่มี Pneumonia อาจมีโรคอื่น ไม่เท่ากับ “healthy”

เริ่มที่ [readme.txt](readme.txt) สำหรับคู่มือภาษาไทยแบบละเอียด
และ [รายงาน Word](report_th.docx) สำหรับรายงานวิธีการ เหตุผล สูตร และการเชื่อมกับบทเรียน 6 บท
มี [ฉบับข้อความต้นฉบับ](report_th.txt) สำหรับค้นหาและสร้างเอกสารใหม่

- [Notebook สำหรับ Google Colab](Chest_Xray_Pneumonia_Colab.ipynb)
- `src/`: data audit, patient split, models, training/resume, metrics, plots, Grad-CAM
- `tests/`: ทดสอบด้วยข้อมูลสังเคราะห์ ไม่ใช่ผลประเมินทางการแพทย์
- `requirements.txt`: dependency ranges; notebook เก็บรุ่นจริงใน environment_freeze.txt

ยังไม่มีผลฝึกจาก dataset จริงหรือ checkpoint ที่ผ่านการประเมิน คะแนนจะถูกสร้างเมื่อผู้ใช้รัน notebook
ใช้เพื่อการศึกษาเท่านั้น

Data provider: **NIH Clinical Center**.
[NIH download](https://nihcc.app.box.com/v/ChestXray-NIHCC) ·
[Kaggle NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data) ·
[Wang et al., CVPR 2017, ChestX-ray8 (ChestX-ray14 in appendix)](https://arxiv.org/abs/1705.02315)
