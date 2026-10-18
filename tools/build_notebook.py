"""Rebuild the Colab notebook without executing any training."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md('''# Chest X-ray: Pneumonia Screening — โปรเจกต์ Deep Learning ปี 3
**Dataset: NIH ChestX-ray14 ตามตัวเลือกของอาจารย์**

งานที่ทำได้จริง: ทำนายว่าภาพมี **label Pneumonia** หรือไม่ โดยกลุ่มลบอาจมีโรคอื่น
ไม่มี COVID-19 ใน label ชุดนี้ จึงไม่ฝึกหรืออ้างผล COVID-19
เป็นการทดลองเพื่อการศึกษา ไม่ใช่เครื่องมือวินิจฉัย

Notebook นี้ยังไม่ได้ฝึกกับข้อมูลจริง ไม่มีคะแนนที่แต่งขึ้น: รันเพื่อสร้างผลของคุณเอง
อ่าน `report_th.txt` สำหรับเหตุผล สูตร และการเชื่อมกับบทเรียนทั้ง 6 บท

**ลำดับ**: ตั้ง GPU → ติดตั้ง/เปิดโค้ด → เลือกที่เก็บ checkpoint → ข้อมูล → audit/split →
ดูภาพ → ทดลอง 3 โมเดล → ล็อกตัวเลือกด้วย validation → เปิด test ครั้งสุดท้าย → อธิบายภาพ

Colab: Runtime → Change runtime type → GPU (ถ้ามีให้เลือก T4 ใช้ได้)
ทรัพยากรฟรีไม่รับประกัน; อ่าน [Colab FAQ](https://research.google.com/colaboratory/faq.html)
''')
md('''## 1. เตรียมโค้ดและไลบรารี
เปิด notebook จาก GitHub หรือ upload ไฟล์นี้ได้ โค้ดจะ clone repository เมื่อยังไม่มี `src/`
หากต้องการใช้ commit เดิมให้ใส่ CODE_REF เป็น commit SHA ที่บันทึกไว้
คง torch/torchvision ที่ Colab จัดมาเป็นคู่ไว้ ไม่ติดตั้ง CUDA ทับโดยไม่จำเป็น
''')
code('''from pathlib import Path
import subprocess, sys, os
REPO = "https://github.com/baconkak/Chest-X-Ray-Pneumonia-COVID-19-Screening.git"
CODE_REF = "main"  # For reproducibility, replace with the saved commit SHA.
if not Path("src/engine.py").exists():
    project = Path("/content/chest-xray-project")
    if not project.exists():
        subprocess.run(["git", "clone", REPO, str(project)], check=True)
    os.chdir(project)
    subprocess.run(["git", "checkout", CODE_REF], check=True)
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "scikit-learn>=1.5,<2", "pandas>=2.2,<4", "matplotlib>=3.8,<4",
                "Pillow>=10.4,<13", "kagglehub>=0.3,<1", "tqdm>=4.66,<5"], check=True)
import torch
from src.engine import environment
print(environment())
print("GPU available:", torch.cuda.is_available())
CODE_SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("Code commit:", CODE_SHA)
''')
md('''## 2. เก็บผลใน Drive และตั้งการทดลอง
อนุญาต Drive เมื่อ Colab ถาม เพื่อเก็บ checkpoint ข้าม session
**quick**: grouped split 70/15/15 ตามผู้ป่วย ใช้ผู้ป่วยสุ่มไม่เกิน 3,000 คนจากข้อมูลที่มี
การจำกัดผู้ป่วยลดเวลาฝึก แต่ไม่ได้ลดไฟล์ที่ Kagglehub ดาวน์โหลด
**full**: ใช้ข้อมูลครบและ official test list; แบ่ง validation 20% ของ development ตามผู้ป่วย

ค่าเริ่มต้นเป็น quick เพื่อทดสอบระบบก่อน ผล subset เทียบตรงกับ paper ไม่ได้
เริ่มงานใหม่ให้เปลี่ยน RUN_NAME; กลับมางานเดิมให้ใช้ชื่อเดิมและ RESUME=True
resume ใช้ config/split เดิมทั้งหมด ห้ามเพิ่ม epochs หรือเปลี่ยน hyperparameter ในงานเดิม
''')
code('''USE_DRIVE = True
PROFILE = "quick"  # "quick" or "full"
RUN_NAME = "nih_quick_seed42_v1"
RESUME = False
if PROFILE not in {"quick", "full"}:
    raise ValueError("PROFILE must be quick or full")
if USE_DRIVE:
    try:
        from google.colab import drive
    except ImportError:
        raise RuntimeError("Outside Colab: set USE_DRIVE=False")
    drive.mount("/content/drive")
OUTPUT_BASE = Path("/content/drive/MyDrive/ChestXrayProject") if USE_DRIVE else Path("outputs")
OUT = OUTPUT_BASE / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
from src.engine import TrainConfig, save_json
CFG = TrainConfig(seed=42, epochs=8, head_epochs=2, batch_size=32)
SPLIT_MODE = "grouped" if PROFILE == "quick" else "official"
MAX_PATIENTS = 3000 if PROFILE == "quick" else None
ALLOW_PARTIAL = PROFILE == "quick"
MODELS = ["mlp", "small_cnn", "resnet18"]
print(OUT)
''')
md('''## 3. ดาวน์โหลด NIH ChestX-ray14 จาก Kaggle หรือใช้ไฟล์ที่แตกไว้แล้ว
แหล่งที่อาจารย์ระบุ: [NIH Chest X-rays บน Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
หรือ [NIH ต้นฉบับ](https://nihcc.app.box.com/v/ChestXray-NIHCC)

ตัวเลือก A: ตั้ง DOWNLOAD_DATA=True ใช้ Kagglehub ดาวน์โหลด dataset (ขนาดหลายสิบ GB)
ตรวจพื้นที่ว่างก่อน; ค่า 70 GB เป็นเกณฑ์เริ่มต้นโดยประมาณ ไม่รับประกันว่าจะพอทุก runtime
ถ้า authentication ถูกขอ ให้ใช้ `kagglehub.login()` ตามคู่มือ Kaggle; ห้ามใส่ token ลง notebook

ตัวเลือก B: ดาวน์โหลดเองจากแหล่งข้างต้น แตก zip/tar **ก่อน** แล้วตั้ง DATA_ROOT ให้ชี้โฟลเดอร์รวม
ต้องมี Data_Entry_2017.csv (หรือชื่อ *_v2020.csv) และ PNG ซึ่งอยู่ใน subfolder ได้
full ต้องมี train_val_list.txt และ test_list.txt และภาพครบทั้งหมด
quick ยอมรับภาพบาง archive ได้ แต่ต้องรายงานว่าเป็น subset และตรวจ class support
ถ้าดึงจาก Drive การอ่านภาพทีละไฟล์อาจช้า คัดลอกข้อมูลลง `/content` เมื่อมีพื้นที่เพียงพอ
''')
code('''import shutil
DOWNLOAD_DATA = True
DATA_ROOT = Path("/content/nih-chestxray14")  # Used when DOWNLOAD_DATA=False
KAGGLE_HANDLE = "nih-chest-xrays/data"  # Record resolved version path below.
if DOWNLOAD_DATA:
    import kagglehub
    free_gb = shutil.disk_usage("/content" if Path("/content").exists() else ".").free / 2**30
    print(f"Free disk: {free_gb:.1f} GiB")
    if free_gb < 70:
        raise RuntimeError("Less than 70 GiB free: use an extracted subset with DOWNLOAD_DATA=False or Kaggle Notebook")
    DATA_ROOT = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
if not DATA_ROOT.is_dir():
    raise FileNotFoundError(f"Set DATA_ROOT to the extracted NIH dataset: {DATA_ROOT}")
print("Resolved dataset path/version:", DATA_ROOT)
''')
md('''## 4. ตรวจ metadata และแบ่งผู้ป่วยก่อน augmentation
ภาพที่มี `Pneumonia|Effusion` เป็น positive ด้วย ไม่ทิ้งภาพหลายโรค
label ไม่พบ Pneumonia = negative ของ label นี้ ไม่ได้แปลว่าสุขภาพดีหรือไม่มี COVID-19
patient stratum คือมีอย่างน้อยหนึ่งภาพ positive ใช้เพื่อรักษาสัดส่วนผู้ป่วยคร่าว ๆ
จำนวนภาพและ prevalence ไม่จำเป็นต้องเท่ากันทุก split

อ่าน summary แล้วตรวจว่าแต่ละชุดมี positive เพียงพอ หากมีเพียงหลักหน่วยควรเพิ่มข้อมูล
ตรวจไฟล์เสียทั้งหมดก่อนฝึก (ใช้เวลาสักพัก) และตรวจตัวอย่างภาพด้วยตา
''')
code('''from src.data import read_metadata, make_splits, save_splits
from PIL import Image
import pandas as pd
df, audit = read_metadata(DATA_ROOT, allow_partial=ALLOW_PARTIAL)
splits = make_splits(df, DATA_ROOT, mode=SPLIT_MODE, seed=CFG.seed, max_patients=MAX_PATIENTS)
if (OUT / "data_audit.json").exists():
    import json
    old_audit = json.loads((OUT / "data_audit.json").read_text(encoding="utf-8"))
    if not RESUME:
        raise FileExistsError("RUN_NAME already exists: use RESUME=True or a new RUN_NAME")
    if old_audit["audit"]["metadata_sha256"] != audit["metadata_sha256"]:
        raise ValueError("Dataset metadata changed; start a new run")
    for split_name, frame in splits.items():
        old = pd.read_csv(OUT / "splits" / f"{split_name}.csv", dtype={"patient_id":str})
        if list(old["Image Index"]) != list(frame["Image Index"]):
            raise ValueError("Split changed; start a new run")
split_hash, summary = save_splits(splits, OUT / "splits")
save_json(dict(audit=audit, summary=summary, split_hash=split_hash,
               protocol=SPLIT_MODE, profile=PROFILE, max_patients=MAX_PATIENTS,
               code_commit=CODE_SHA, dataset_handle=KAGGLE_HANDLE,
               resolved_dataset_path=str(DATA_ROOT)), OUT / "data_audit.json")
display(pd.DataFrame(summary).T)
for split_name, frame in splits.items():
    for path in frame.path:
        with Image.open(path) as im:
            im.verify()
print("Image integrity and patient overlap checks passed; split hash:", split_hash)
''')
md('''## 5. ดูภาพ training และ augmentation
resize แบบ padding รักษาขอบเขตภาพ ไม่มี crop สุ่ม และไม่กลับซ้ายขวา
ใช้หมุนไม่เกิน 5 องศา / เลื่อน 2% เฉพาะ training — ยังต้องตรวจว่าเหมาะกับภาพจริง
ImageNet normalization เป็นค่าคงที่จาก pretraining ไม่ได้ fit จาก validation/test
ทำ grayscale เป็น RGB ซ้ำเพื่อใช้ pretrained backbone
''')
code('''import matplotlib.pyplot as plt
import numpy as np
from src.data import ChestDataset
sample = splits["train"].groupby("target", group_keys=False).head(3)
dataset = ChestDataset(sample, train=True, size=CFG.image_size)
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for i, ax in enumerate(axes.flat):
    if i >= len(dataset):
        ax.axis("off"); continue
    x, y = dataset[i]
    image = x.permute(1,2,0).numpy() * [.229,.224,.225] + [.485,.456,.406]
    ax.imshow(np.clip(image, 0, 1)); ax.set_title(f"Pneumonia label={int(y.item())}"); ax.axis("off")
plt.tight_layout(); plt.show()
''')
md('''## 6. ฝึก MLP → Small CNN → ResNet18
MLP ใช้ภาพลดเหลือ 64×64 หลัง pipeline เดียวกัน เป็นตัวเทียบกับบท feedforward
CNN ใช้ locality/weight sharing; ResNet18 เริ่มจาก ImageNet และมี residual connections
ฝึก ResNet head 2 epochs แล้วปลด layer4 ใช้ learning rate ลดลง; BN คงค่า pretrained

Loss = **BCEWithLogitsLoss** (หนึ่ง logit, target float [B,1]) ตามบท 3 ไม่ใส่ sigmoid ซ้ำ
pos_weight = จำนวนลบ/จำนวนบวกจาก **training เท่านั้น**; ไม่ oversample ซ้ำ
AdamW/weight decay, dropout, gradient clipping และ early stopping ช่วยควบคุมการฝึก
เลือก checkpoint จาก validation AP สูงสุด (min_delta=0.0001) และ patience=3 ใน fine-tune
ไม่ตัดสินด้วย test ระหว่างฝึก

train loss แบบ weighted กับ validation BCE แบบ unweighted มีวัตถุประสงค์ต่างกัน
จึงไม่อ่านระยะห่างของสองค่าเป็น generalization gap โดยตรง
checkpoint เก็บทุก epoch รวม optimizer/scheduler/AMP/RNG; การหลุดกลาง epoch จะเริ่ม epoch นั้นใหม่
''')
code('''from src.engine import train_model, load_trained
from src.plots import history_plot
validation_rows = []
for name in MODELS:
    resume_this = RESUME and (OUT / name / "last.pt").exists()
    model, validation, history = train_model(name, splits, CFG, split_hash, OUT, resume=resume_this)
    validation_rows.append(dict(model=name, **validation))
    history_plot(history, OUT / name / "learning_curves.png")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
validation_table = pd.DataFrame(validation_rows)
validation_table.to_csv(OUT / "validation_comparison.csv", index=False)
display(validation_table)
''')
md('''## 7. ล็อกวิธีที่เลือก ก่อนเปิด test
อันดับหลักคือ validation Average Precision (AP ของ scikit-learn; ไม่ใช่ trapezoidal PR-AUC)
แต่ละโมเดลใช้ threshold ที่เลือกจาก validation: precision สูงสุดภายใต้ recall ≥ 0.90
ถ้าเสมอเลือก threshold สูงสุด ค่า 0.90 เป็นเป้าหมายการทดลอง ไม่ใช่มาตรฐานทางคลินิก
test recall อาจต่ำกว่า 0.90 ได้; threshold ไม่รับประกันผลบนผู้ป่วยใหม่

เลือกโมเดลด้วย validation เท่านั้น จากนั้นบันทึกตัวเลือกและประเมิน test
baseline คง score เท่ากับ prevalence ของ training; hard decision เป็น majority class ของ training
''')
code('''import json
winner = validation_table.sort_values("ap", ascending=False, kind="stable").iloc[0]["model"]
selection = dict(selected_model=winner, primary_metric="validation AP",
                 split_hash=split_hash, model_order=MODELS,
                 tie_break="first in declared model order", config=CFG.__dict__,
                 thresholds={row["model"]: row["threshold"] for row in validation_rows})
save_json(selection, OUT / "locked_selection.json")
print("Locked model:", winner)
''')
md('''## 8. Final test — รันหลังปิดการปรับโมเดลเท่านั้น
ตั้ง RUN_FINAL_TEST=True เมื่อยอมรับ protocol/config แล้ว ผลทดสอบใช้เพื่อรายงานครั้งสุดท้าย
ถ้าเปลี่ยนโมเดลจากผล test ชุดนี้จะกลายเป็น development data ต้องมี holdout ใหม่
รายงานทุกโมเดลตามแผนที่กำหนด รวม majority baseline ไม่เลือกผู้ชนะใหม่จาก test
bootstrap สุ่มทั้งผู้ป่วย 200 ครั้ง เพื่อประมาณ finite-sample uncertainty ของ image-level AP
ไม่ใช่ความไม่แน่นอนรายบุคคลหรือความแปรปรวนจากการฝึกหลาย seeds
''')
code('''RUN_FINAL_TEST = False  # Set True only after development choices are frozen.
if RUN_FINAL_TEST:
    from torch.utils.data import DataLoader
    from src.engine import predict
    from src.metrics import binary_metrics, patient_bootstrap_ap
    from src.plots import evaluation_plot
    test_dir = OUT / "final_test"
    if (test_dir / "test_comparison.csv").exists():
        raise RuntimeError("Final test already recorded. Read saved results; do not tune against test.")
    test_dir.mkdir(exist_ok=True)
    test_frame = splits["test"]
    test_loader = DataLoader(ChestDataset(test_frame, size=CFG.image_size), batch_size=CFG.batch_size)
    rows, predictions = [], test_frame[["Image Index", "patient_id", "target"]].copy()
    for name in MODELS:
        model, _ = load_trained(OUT / name / "best.pt", "cuda" if torch.cuda.is_available() else "cpu")
        y, scores = predict(model, test_loader, next(model.parameters()).device)
        threshold = selection["thresholds"][name]
        metrics = binary_metrics(y, scores, threshold)
        predictions[name] = scores
        boot = test_frame[["patient_id", "target"]].copy(); boot["score"] = scores
        ci = patient_bootstrap_ap(boot, repeats=200)
        rows.append(dict(model=name, **metrics, ap_ci_low=ci["low"], ap_ci_high=ci["high"],
                         bootstrap_valid=ci["valid_replicates"]))
        counts = evaluation_plot(y, scores, metrics, test_dir / f"{name}_evaluation.png")
        bin_ids = np.minimum((scores * 10).astype(int), 9)
        pd.DataFrame([dict(bin=i, left=i/10, right=(i+1)/10, n=int(counts[i]),
                           mean_score=float(scores[bin_ids==i].mean()) if counts[i] else None,
                           observed_fraction=float(y[bin_ids==i].mean()) if counts[i] else None)
                      for i in range(10)]).to_csv(test_dir / f"{name}_calibration.csv", index=False)
        # Error review is descriptive only; do not retune after inspecting.
        errors = test_frame.copy(); errors["score"] = scores; errors["prediction"] = (scores >= threshold).astype(int)
        errors["error"] = np.select([(errors.target==1)&(errors.prediction==0),
                                      (errors.target==0)&(errors.prediction==1)], ["FN","FP"], default="correct")
        errors.drop(columns="path").to_csv(test_dir / f"{name}_errors.csv", index=False)
        if "View Position" in errors.columns:
            slices = []
            for view, frame in errors.groupby("View Position", dropna=False):
                slices.append(dict(view=str(view), **binary_metrics(frame.target, frame.score, threshold)))
            pd.DataFrame(slices).to_csv(test_dir / f"{name}_view_slices.csv", index=False)
        del model
    prevalence = float(splits["train"].target.mean())
    rows.append(dict(model="majority_train_prevalence", **binary_metrics(y, np.full(len(y), prevalence), .5)))
    predictions.to_csv(test_dir / "all_predictions.csv", index=False)
    results = pd.DataFrame(rows); results.to_csv(test_dir / "test_comparison.csv", index=False)
    delta = patient_bootstrap_ap(predictions, score_a="mlp", score_b=winner, repeats=200)
    save_json(dict(comparison=f"{winner} minus mlp AP", paired_patient_bootstrap=delta), test_dir / "paired_ap_delta.json")
    display(results)
else:
    print("Test remains unopened. Review validation results and set RUN_FINAL_TEST=True when ready.")
''')
md('''## 9. Grad-CAM และทดลองภาพเดี่ยว
Grad-CAM แสดงบริเวณที่สัมพันธ์กับ positive logit ของ ResNet18 ไม่ใช่ segmentation หรือหลักฐานยืนยันโรค
ภาพเดี่ยวต้องเป็น PNG/JPEG chest X-ray ที่สอดคล้องกับข้อมูลฝึก ไม่ใช้ DICOM โดยตรง
score จาก weighted loss ยังไม่ใช่ความน่าจะเป็นทางคลินิกที่สอบเทียบแล้ว
''')
code('''IMAGE_PATH = ""  # Set an image path to run the example. No image is uploaded anywhere.
if IMAGE_PATH:
    from src.data import transforms
    from src.explain import grad_cam_resnet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_trained(OUT / "resnet18" / "best.pt", device)
    with Image.open(IMAGE_PATH) as image:
        x = transforms(size=CFG.image_size)(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        score = float(model(x).sigmoid().item())
    threshold = selection["thresholds"]["resnet18"]
    print({"pneumonia_label_score": score, "threshold": threshold,
           "flag_for_educational_review": score >= threshold, "covid19": "not supported"})
    cam = grad_cam_resnet(model, x)[0]
    rgb = x[0].detach().cpu().permute(1,2,0).numpy() * [.229,.224,.225] + [.485,.456,.406]
    plt.imshow(np.clip(rgb,0,1)); plt.imshow(cam, cmap="jet", alpha=.4)
    plt.axis("off"); plt.title("Positive-logit Grad-CAM (not a diagnosis)"); plt.show()
''')
md('''## 10. เก็บหลักฐานและเขียนผลจริงลงรายงาน
ไฟล์ `data_audit.json`, `splits/`, `run.json`, `history.csv`, `validation_comparison.csv`,
`locked_selection.json`, `final_test/` และ checkpoints อยู่ใน OUT
เก็บ `pip freeze`, commit, เวลาฝึก, GPU และรูป learning curves ประกอบรายงาน
อย่า push dataset, token หรือ weights เข้า Git; `.gitignore` มีให้แล้ว

ทดลองเพิ่มเมื่อมีเวลา: unweighted BCE vs weighted BCE, ResNet random initialization vs pretrained,
และ seed 43/44 โดยใช้ **split เดิม** (แยก seed split ออกจาก seed training) ก่อนเปิด final test
สร้าง run แยกและบันทึกแผน อย่าเลือกจาก test แล้ววนปรับ
อ่านหัวข้อผลทดลองใน `report_th.txt` ซึ่งเว้นไว้สำหรับคะแนนจริง

อ้างอิง: NIH Clinical Center เป็นผู้ให้ข้อมูล; Wang et al., CVPR 2017,
[ChestX-ray8 / ChestX-ray14 appendix](https://arxiv.org/abs/1705.02315),
[NIH download](https://nihcc.app.box.com/v/ChestXray-NIHCC),
[PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial),
[BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
''')
code('''(OUT / "environment_freeze.txt").write_text(
    subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
print("Artifacts saved in:", OUT)
print("Use the result files to complete the report; do not report smoke-test scores as real performance.")
''')

nb.cells = cells
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python'},
               'colab': {'name': 'Chest_Xray_Pneumonia_Colab.ipynb'}, 'accelerator': 'GPU'}
nbf.validate(nb)
nbf.write(nb, ROOT / 'Chest_Xray_Pneumonia_Colab.ipynb')
print(f'Built {len(cells)} cells')
