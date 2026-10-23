"""Validation review first, followed by controlled, low-LR unfreezing."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook()
sources=[('markdown','''# ตรวจภาพผิดและทดลองปลด backbone
ใช้ข้อมูลและ validation เดิมจาก `nih_full_resnet50_v3` ไม่สุ่มแบ่งใหม่ ไม่เปิด final test
เริ่มจาก ResNet18 best.pt เดียวกันทุก arm และเก็บผลแยกที่ `nih_backbone_diagnostics_v1`
ขั้นแรกตรวจภาพต้นฉบับ เทียบ input 224 px และ Grad-CAM พร้อม metadata โดยไม่แก้ label อัตโนมัติ
แยก FP/FN ตาม NIH label ไม่ใช่การยืนยันทางการแพทย์

ควรรัน cell เหล่านี้ต่อท้าย notebook เดิมใน runtime ที่ยังมีข้อมูล/cache อยู่
runtime ใหม่ที่ไม่มีข้อมูลจะหยุดเพื่อป้องกันโหลดต้นฉบับ 42 GB ซ้ำโดยไม่ตั้งใจ
ใช้ L4 เมื่อเริ่มฝึก; หยุดก่อน training ได้หากต้องการตรวจภาพอย่างเดียว
'''),('code','''from pathlib import Path
import os, sys, subprocess, json
project=Path('/content/chest-xray-improved')
if not (project/'.git').is_dir():
    raise RuntimeError('Use the existing Colab runtime with the project and NIH images first')
os.chdir(project)
subprocess.run(['git','fetch','origin','main'],check=True)
subprocess.run(['git','checkout','--detach','FETCH_HEAD'],check=True)
sys.path.insert(0,str(project))
for key in list(sys.modules):
    if key=='src' or key.startswith('src.'):del sys.modules[key]
from src.data import read_metadata, prepare_image_cache
from src.diagnostics import restore_splits, validation_errors, render_review, run_backbone_trials
SOURCE_RUN=Path('/content/drive/MyDrive/ChestXrayProject/nih_full_resnet50_v3')
DIAG_OUT=SOURCE_RUN.parent/'nih_backbone_diagnostics_v1'
RESUME_DIAGNOSTICS=False
if not (SOURCE_RUN/'data_audit.json').exists():
    raise RuntimeError('Mount the same Google Drive account and set SOURCE_RUN')
audit=json.loads((SOURCE_RUN/'data_audit.json').read_text())
DATA_ROOT=Path(audit['resolved_dataset_path'])
if not DATA_ROOT.is_dir():
    raise RuntimeError('Original NIH images are not in this runtime; no automatic download was started')
metadata,current_audit=read_metadata(DATA_ROOT)
if current_audit['metadata_sha256']!=audit['audit']['metadata_sha256']:
    raise ValueError('Metadata changed')
frozen_splits,frozen_hash=restore_splits(SOURCE_RUN,metadata)
errors,checkpoint,source_selection=validation_errors(SOURCE_RUN,frozen_splits['val'],DIAG_OUT/'error_review')
print('Exact split hash:',frozen_hash)
print('Validation errors:',errors.error.value_counts().to_dict())
print('Output:',DIAG_OUT)
'''),('markdown','''## ตรวจภาพก่อนฝึก
เลือก FN score ต่ำ, FP score สูง, FP สุ่ม และ TP/TN สำหรับเปรียบเทียบ โดยไม่ซ้ำผู้ป่วยภายในแต่ละกลุ่ม
รายชื่อข้อผิดพลาดทั้งหมดอยู่ใน validation_errors.csv; ภาพที่เลือกดูไม่ใช่ตัวแทนเชิงสถิติของทั้งชุด
ตรวจขอบภาพ ตัวอักษร ท่อ/อุปกรณ์ การย่อภาพ และบริเวณที่ CAM เน้น
CAM แสดงความสัมพันธ์กับ score ไม่ได้ยืนยันโรคหรือพิสูจน์สาเหตุเชิงเหตุผล
review_annotations.csv เว้นช่องให้บันทึกข้อสังเกต/คำถามสำหรับผู้เชี่ยวชาญ ไม่ relabel ภาพเอง
'''),('code','''from IPython.display import display, Image as DisplayImage
review_images=render_review(errors,checkpoint,DIAG_OUT/'error_review')
for path in review_images:
    print(path.name)
    display(DisplayImage(filename=str(path)))
'''),('markdown','''## ทดลองปลด backbone เพิ่ม
เทียบ control_layer4, progressive_layer34 และ progressive_all ด้วยจุดเริ่มและ seed เดียวกัน
ทุก arm ฝึก 9 epochs แบ่ง 3 ช่วง ช่วงละ 3 epochs; 3 epochs แรกเปิด layer4 เหมือนกัน
ช่วงถัดไป control คง layer4; อีกสอง arm เปิด layer3+4; ช่วงสุดท้ายเปิดทุกชั้นเฉพาะ progressive_all
หัว LR=1e-5, layer4=3e-6, layer3=9e-7 และชั้นก่อนหน้าต่ำลงอีก 0.3 เท่าต่อช่วงชั้น
BatchNorm running statistics คงที่ในทุก arm เพื่อไม่เปลี่ยนสองปัจจัยพร้อมกัน
optimizer เก็บ Adam moments ข้ามช่วง; reset scheduler/LR ที่ขอบช่วงเหมือนกันทุก arm
loss weight=10, batch=64, input224, seed42 และ validation เหมือนเดิม
ต้นทุนเพิ่มราว 27 epochs; เวลา full-backbone ต้องวัดจริง ไม่รับประกันผลดีขึ้น
ถ้าไม่มี epoch ใหม่ดีกว่าจุดเริ่ม จะเก็บ baseline เดิมเป็น best (selected_epoch=0)
'''),('code','''import pandas as pd
import torch
from src.engine import save_json
if not torch.cuda.is_available():raise RuntimeError('Choose L4 GPU before training')
# Reuse/create only the deterministic input cache; original paths are kept for review.
combined=pd.concat(list(frozen_splits.values()),ignore_index=True)
cached=prepare_image_cache(combined,Path('/content/nih_cache'),224,workers=4)
path_map=cached.set_index('Image Index').path
training_splits={k:f.assign(path=f['Image Index'].map(path_map)) for k,f in frozen_splits.items()}
comparison=run_backbone_trials(training_splits,frozen_hash,checkpoint,errors,DIAG_OUT,
                              resume=RESUME_DIAGNOSTICS)
save_json(dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               source_run=str(SOURCE_RUN),split_hash=frozen_hash),DIAG_OUT/'provenance.json')
display(comparison)
print('Validation only. Do not choose or tune against final test.')
''')]
nb.cells=[(nbf.v4.new_code_cell(text) if kind=='code' else nbf.v4.new_markdown_cell(text))
          for kind,text in sources]
for i,c in enumerate(nb.cells):c.id=f'diagnostic-{i:02d}'
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
             'language_info':{'name':'python'},'accelerator':'GPU',
             'colab':{'name':'Chest_Xray_Backbone_Diagnostics.ipynb'}}
nbf.validate(nb)
nbf.write(nb,ROOT/'Chest_Xray_Backbone_Diagnostics.ipynb')
print('Built diagnostic notebook')
