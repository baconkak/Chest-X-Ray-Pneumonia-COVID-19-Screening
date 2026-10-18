คู่มือใช้งาน Chest X-ray Pneumonia Screening
สำหรับวิชา Deep Learning ปี 3 เทอม 1 วิศวกรรมคอมพิวเตอร์
===========================================================

1. สิ่งที่ได้รับและสิ่งที่ต้องทำต่อ

โปรเจกต์นี้เตรียม pipeline สำหรับฝึกจริงบน Google Colab แล้ว
มี MLP ตามเนื้อหา feedforward, CNN ขนาดเล็ก และ ResNet18 แบบ transfer learning
มีรายงาน Word อธิบายเหตุผลใน report_th.docx แต่ยังไม่มีคะแนนทดลองจริง
ผู้ใช้ต้องดาวน์โหลดข้อมูล เลือก GPU และรัน notebook เพื่อสร้างผลของตัวเอง
อย่านำผล synthetic tests ไปอ้างว่าเป็นประสิทธิภาพตรวจโรค

ข้อจำกัดสำคัญ: NIH ChestX-ray14 ไม่มี label COVID-19
งานที่ทำคือ Pneumonia label present / absent เท่านั้น
negative อาจมีโรคอื่น; No Finding ไม่ใช่ใบรับรองว่าปอดปกติ
ชื่อ repo ตามชื่อหัวข้อเดิม ไม่ได้หมายถึงโมเดลรองรับ COVID-19 แล้ว

2. โครงสร้างไฟล์

Chest_Xray_Pneumonia_Colab.ipynb : จุดเริ่มต้น ใช้งานตามลำดับ cell
report_th.docx                 : รายงาน Word จัดหน้า พร้อมสูตรและตารางสำหรับอ่านและแก้ไข
report_th.txt                  : ข้อความต้นฉบับของรายงาน
readme.txt                     : คู่มือฉบับนี้
README.md                      : หน้าแรก GitHub และปุ่มเปิด Colab
requirements.txt               : dependencies สำหรับเครื่องส่วนตัว
src/data.py                    : ตรวจข้อมูล, label, patient split, preprocessing
src/models.py                  : MLP, SmallCNN, ResNet18
src/engine.py                  : ฝึก, checkpoint, resume, inference
src/metrics.py                 : AP, AUROC, threshold และ bootstrap
src/plots.py                   : กราฟการฝึก, PR, ROC, calibration, confusion matrix
src/explain.py                 : Grad-CAM สำหรับ ResNet18
tests/test_pipeline.py         : ทดสอบด้วยภาพสังเคราะห์
tools/build_notebook.py        : สร้าง notebook ใหม่หลังแก้เนื้อหา notebook
.gitignore                    : กัน dataset, checkpoints, tokens และไฟล์ชั่วคราว

3. เปิด notebook บน Colab

3.1 เข้า GitHub repository:
https://github.com/baconkak/Chest-X-Ray-Pneumonia-COVID-19-Screening
กดปุ่ม Open in Colab ใน README หรือเปิดลิงก์:
https://colab.research.google.com/github/baconkak/Chest-X-Ray-Pneumonia-COVID-19-Screening/blob/main/Chest_Xray_Pneumonia_Colab.ipynb
อีกวิธี: เข้า https://colab.research.google.com แล้ว File > Upload notebook
เลือก Chest_Xray_Pneumonia_Colab.ipynb จากเครื่อง

3.2 File > Save a copy in Drive เพื่อเก็บฉบับของตัวเอง
3.3 Runtime > Change runtime type > เลือก GPU ถ้ามี T4 ก็เพียงพอสำหรับเริ่มทดลอง
Colab ฟรีไม่รับประกันชนิด GPU ระยะเวลา หรือโควตา อย่าวางแผนว่าต้องได้ GPU เสมอ
3.4 รัน cell ทีละส่วนจากบนลงล่าง อ่านผล audit และตัวอย่างภาพก่อนเริ่มฝึก
cell แรกจะนำ source จาก GitHub มาใช้ และพิมพ์ commit SHA ให้จดไว้ในรายงาน
หากแก้โค้ดในเครื่องต้อง push ก่อน หรือ upload src/ ฉบับที่แก้เข้า runtime
หากรันจากโฟลเดอร์ repo ที่มี src/ แล้ว notebook จะใช้โค้ดในโฟลเดอร์นั้น

4. เลือกที่เก็บผลและขนาดงาน

USE_DRIVE=True จะขออนุญาต mount Google Drive ใน Colab
ผลอยู่ที่ MyDrive/ChestXrayProject/<RUN_NAME>/ ไม่ต้องเก็บ dataset ลง Git
USE_DRIVE=False จะเก็บใน outputs/ ของ runtime ซึ่งหายได้เมื่อ runtime ถูกลบ

PROFILE="quick": เลือกผู้ป่วยสุ่มไม่เกิน 3,000 คน แบ่ง grouped 70/15/15
ใช้สำหรับทดลองให้ระบบทำงานก่อน ไม่ใช่ผล official benchmark
การจำกัดผู้ป่วยไม่ลดขนาดดาวน์โหลด Kagglehub ซึ่งยังดึง dataset เต็ม
PROFILE="full": ข้อมูลครบ, official test list, แบ่ง val 20% ของ train_val ตามผู้ป่วย
เปลี่ยน RUN_NAME ด้วยเมื่อเปลี่ยน profile หรือ config

ค่าเริ่มต้น: epochs=8, head_epochs=2, batch=32, image_size=224, seed=42
เปลี่ยนได้ใน TrainConfig ก่อนเริ่มงานใหม่
ถ้า GPU memory ไม่พอ ให้ลด batch_size เป็น 16 หรือ 8 และใช้ชื่อ run ใหม่
หากต้องทดลองหลาย seed บน split เดิม ให้คง seed ใน make_splits ไว้ 42
แต่เปลี่ยน CFG.seed เป็น 43/44 หลังสร้าง split (แยก seed การแบ่งกับ seed การฝึก)
บันทึก split hash เดิมเพื่อยืนยันว่าเทียบข้อมูลชุดเดียวกัน

5. เตรียม dataset ตามที่อาจารย์เลือก

แหล่งหลัก:
https://www.kaggle.com/datasets/nih-chest-xrays/data
แหล่งต้นฉบับ NIH:
https://nihcc.app.box.com/v/ChestXray-NIHCC
ผู้ให้ข้อมูลคือ NIH Clinical Center; ดูข้อกำหนด attribution ใน DATA_SOURCES.txt

วิธี A (อัตโนมัติ): DOWNLOAD_DATA=True ใน notebook
ใช้ kagglehub.dataset_download("nih-chest-xrays/data") และบันทึก path/version ที่ได้
ข้อมูลขนาดหลายสิบ GB ต้องมีพื้นที่เหลือสำหรับ cache และผลการฝึก
cell จะหยุดถ้าพื้นที่ว่างต่ำกว่า 70 GiB ซึ่งเป็นเพียงเกณฑ์เริ่มต้นโดยประมาณ
ถ้า dataset อยู่ใน cache แล้วแต่เหลือพื้นที่ไม่ถึงเกณฑ์ ให้ใช้วิธี B ชี้ path cache เดิม
หาก Kaggle ขอ login ให้ทำตาม https://github.com/Kaggle/kagglehub#authenticate
ใช้ kagglehub.login() ใน cell ชั่วคราวหรือระบบ Secrets ที่เหมาะสม ไม่เขียน token ลงไฟล์
ถ้า download ล้มเหลว/มีข้อกำหนดให้ยอมรับ ให้เปิดหน้า dataset ตรวจบัญชีและสิทธิ์ก่อน

วิธี B (มีไฟล์แล้ว): DOWNLOAD_DATA=False
ตั้ง DATA_ROOT=Path("/content/nih-chestxray14") หรือ path ที่มีข้อมูลจริง
ดาวน์โหลดและแตก archive ทุกชั้นก่อนใช้งาน ไม่ให้เหลือแต่ไฟล์ zip/tar
ตัวอย่างโครงสร้าง:
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

ตัวอ่านค้น PNG ใน subfolder ได้ ไม่ต้องย้ายภาพมารวมกัน
ถ้ามี PNG ชื่อซ้ำในหลายโฟลเดอร์จะหยุด เพื่อไม่เลือกภาพผิดเงียบ ๆ
quick ยอมรับบาง archive ได้เมื่อ ALLOW_PARTIAL=True และรายงาน missing images
full จะไม่ยอมรับภาพขาด, official lists ขาด, lists ซ้อนกัน หรือผู้ป่วยข้ามชุด
อย่าใช้ชุด pediatric Pneumonia หรือ COVID dataset อื่นแทนโดยตั้งชื่อว่า NIH

ถ้าข้อมูลเต็มใหญ่เกิน Colab ทางเลือกคือ Kaggle Notebook แล้ว Add Input เป็น dataset เดิม
ตั้ง DOWNLOAD_DATA=False, USE_DRIVE=False และ DATA_ROOT เป็น path ใต้ /kaggle/input ที่เห็นจริง
ตั้ง OUT ให้อยู่ /kaggle/working เพื่อให้เก็บผลได้ สภาพแวดล้อมนี้ต้องเลือก accelerator ตามที่บัญชีมี
ไม่ได้รับประกันโควตาหรือ GPU ของ Kaggle; เช็กในหน้าสร้าง notebook ของบัญชีตนเอง

6. ตรวจข้อมูลก่อนฝึก

cell audit จะพิมพ์ images, patients, positive_images และ prevalence ของทุก split
แต่ละ split ต้องมีทั้งสอง class ถ้า positive น้อยมากให้เพิ่มจำนวนผู้ป่วย/ภาพ
ระบบใช้ Patient ID แยกผู้ป่วยก่อน augmentation
ภาพคนเดียวกันไม่อยู่ทั้ง train/validation/test; ภาพหลายโรคยังเก็บไว้
เช่น Pneumonia|Effusion มี target=1; Effusion อย่างเดียวมี target=0
ตรวจภาพเสียทั้งหมดด้วย Pillow verify และดูตัวอย่างภาพ augmentation
การตรวจชื่อไฟล์ไม่ใช่การตรวจ near-duplicate ทุกชนิด: ดูข้อจำกัดในรายงาน
ไม่ส่งไฟล์ Sheet ที่ใช้เรียนขึ้น GitHub รายงานอ้างชื่อบทและเลขหน้าเท่านั้น

7. การฝึกและการกลับมาฝึกต่อ

ครั้งแรกใช้ RESUME=False และ RUN_NAME ใหม่
MLP, SmallCNN, ResNet18 จะฝึกตามลำดับ เก็บ learning curves และ history
ResNet18 ต้องดาวน์โหลด ImageNet weights ครั้งแรก ต้องมีอินเทอร์เน็ต
ไม่แอบเปลี่ยนเป็น random weights เมื่อ download ไม่ได้
ค่า pretrained=False ใช้ได้สำหรับ ablation แต่ต้องใช้ run แยกและอธิบายว่าไม่ใช่ transfer learning

เมื่อ Colab หลุด:
  1) เชื่อม GPU ใหม่ รัน setup และ mount Drive
  2) ใส่ RUN_NAME เดิม, profile/config เดิม และข้อมูล version เดิม
  3) ตั้ง RESUME=True แล้วรันตั้งแต่ต้นตามลำดับ
  4) ต้องมีทั้ง best.pt และ last.pt ของโมเดลที่ resume
หากหลุดก่อนเริ่มบางโมเดล ระบบจะฝึกโมเดลนั้นจากต้นเมื่อยังไม่มี last.pt
checkpoint เก็บทุกจบ epoch; ถ้าหลุดกลาง epoch งานตั้งแต่ checkpoint ล่าสุดจะถูกทำใหม่
กลับมาฝึกต่อด้วย optimizer, scheduler, scaler และ RNG เดิม ไม่ใช่แค่ load weights
resume ไม่ยอมให้เปลี่ยน config/split; ถ้าต้องการทดลองเพิ่มให้เริ่ม RUN_NAME ใหม่
best.pt เป็นโมเดลที่เลือกจาก validation, last.pt เป็นสถานะล่าสุดสำหรับ resume
โมเดลที่ early stop แล้วจะไม่ฝึกเพิ่มเมื่อ resume

8. การประเมินผล

อ่าน validation_comparison.csv เลือกโมเดลด้วย validation AP เท่านั้น
threshold มาจาก validation: precision สูงสุดภายใต้ recall >= 0.90
tie ใช้ threshold สูงสุด ส่วน tie ของโมเดลใช้ลำดับ MODELS ที่ประกาศไว้
0.90 เป็นเป้าทดลอง ไม่ใช่ข้อรับรองทางการแพทย์
Notebook จะเขียน locked_selection.json ก่อนเปิด test

เมื่อปิดการปรับทุกอย่างแล้ว ตั้ง RUN_FINAL_TEST=True และรันส่วน final test
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

9. ทดลองภาพเดี่ยวและ Grad-CAM

ตั้ง IMAGE_PATH ในส่วนท้ายเป็น path PNG/JPEG แล้วรัน
จะรายงาน pneumonia label score และ threshold ของ ResNet18
score ไม่ได้ผ่าน clinical calibration; ห้ามตีความว่าเป็นโอกาสป่วยจริงของผู้ป่วย
COVID-19 จะถูกแสดงว่า not supported
Grad-CAM เป็นเพียงคำอธิบายการตอบสนองของโมเดล ไม่ใช่ขอบเขตรอยโรค

10. เขียนผลรายงานและส่งงาน

อ่าน report_th.docx แล้วเติมตารางผลจากไฟล์จริง ระบุว่า quick/full และ version ใด
ใส่ชื่อผู้จัดทำ รหัสนักศึกษา รายวิชา ผู้สอน ตามข้อมูลจริง
แนบ learning curves, PR/ROC, confusion matrix, error analysis และข้อจำกัด
อธิบายด้วยว่าผลต่างมีความไม่แน่นอน ไม่สรุปว่า ResNet ดีกว่าเสมอ
หากยังไม่ได้ฝึก ให้ระบุว่ายังไม่มี experimental results ตามความจริง
เก็บ checkpoints และผลใน Drive; ข้อมูลดิบไม่ควรขึ้น Git

11. ใช้ในเครื่องส่วนตัว (ทางเลือก)

ติดตั้ง Python ที่รองรับ dependency และสร้าง environment แยก
ติดตั้ง torch กับ torchvision คู่ที่ตรงกันจาก https://pytorch.org/get-started/locally/
จากนั้นในโฟลเดอร์ Project:
  python -m pip install -r requirements.txt
  python -m unittest discover -s tests -v
เปิด ipynb ด้วย Jupyter/VS Code ตั้ง USE_DRIVE=False และ DOWNLOAD_DATA/DATA_ROOT ตามข้อมูลจริง
tests ใช้ข้อมูลสังเคราะห์เพื่อเช็ก workflow ไม่ต้องมี dataset หรือโหลด pretrained weights
หากแก้ tools/build_notebook.py ให้รัน python tools/build_notebook.py เพื่อปรับ ipynb ให้ตรงกัน

12. ปัญหาที่พบบ่อย

No module named src: รัน cell setup จากต้น และตรวจว่ามี repo/src จริง
torchvision import error: ใช้คู่ torch/torchvision ที่เข้ากัน อย่าอัปเดตทีละตัว
CUDA out of memory: ลด batch, รีสตาร์ต runtime, ใช้ชื่อ run ใหม่ถ้าเปลี่ยน config
No PNG found: แตก archive ซ้อนให้ครบและตั้ง DATA_ROOT ถูกที่
Missing metadata: ใช้ Data_Entry_2017*.csv ของ NIH ไม่ใช่ CSV dataset อื่น
Too few positive patients: เพิ่มข้อมูล/ผู้ป่วย อย่าแก้ด้วยการ split แบบสุ่มรายภาพ
Checkpoint mismatch: ใช้ config และ split เดิม หรือสร้างงานใหม่
Drive ช้า: อ่านภาพจาก local runtime และเก็บเฉพาะ checkpoint/results บน Drive
NaN loss/gradient: หยุดตรวจภาพ/learning rate ไม่แทน NaN ด้วยศูนย์
AMP overflow ช่วงแรก: ระบบข้าม update และลด scale อัตโนมัติ ดู amp_skipped_steps ใน history
ถ้าเกิดซ้ำมากจะหยุด ให้ตรวจข้อมูล/learning rate หรือเริ่ม run ใหม่ด้วย amp=False
Download ต้อง login: เปิด dataset และทำตามระบบของ Kaggle ไม่เก็บ token ใน Git

เอกสาร Colab: https://research.google.com/colaboratory/faq.html
