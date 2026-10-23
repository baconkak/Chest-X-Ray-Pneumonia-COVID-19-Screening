"""Build the larger-model notebook, keeping the original course baseline notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.read(ROOT/'Chest_Xray_Pneumonia_Colab.ipynb', as_version=4)


def cell(index, text):
    nb.cells[index].source = text.strip()
    nb.cells[index].id = f'improved-{index:02d}'


cell(0, '''# NIH Pneumonia — Larger models & improved training (L4 GPU)
รุ่นปรับปรุงหลังอ่านผล quick_seed42: **ResNet50 pretrained** เป็นโมเดลขนาดใหญ่ขึ้น
เทียบ ResNet18 ภายใต้ split/config เดียวกัน และทดลอง unweighted BCE
มี DenseNet121 เป็นตัวเลือก (ลึกกว่าแต่พารามิเตอร์น้อยกว่า ResNet18)

ค่าเริ่มต้นใช้ **ข้อมูลครบ / official test list** ไม่จำกัด 3,000 ผู้ป่วย
ลด LR, cache ภาพ 224 px ลงดิสก์ local, อ่านข้อมูลหลาย workers, AMP และ checkpoint ใน Drive
ผลจริงต้องรอฝึก ไม่มีการอ้างว่าโมเดลใหญ่จะให้คะแนนดีกว่าแน่นอน
NIH ไม่มี COVID-19: ทำนาย **Pneumonia label present / absent** เท่านั้น
ผลลบอาจมีโรคอื่น ใช้เพื่อการศึกษา

Runtime → Change runtime type → **L4 GPU** (ขึ้นกับสิทธิ์/ทรัพยากรบัญชี)
Run all จะฝึกและสรุป validation แล้วจบ; final test ปิดไว้จนกว่าจะยุติการปรับโมเดล
ถ้าหลุดให้เปิดไฟล์เดิม ตั้ง RESUME=True แล้วรันใหม่ด้วย config/ชื่อเดิมทุกค่า
''')
cell(2, '''from pathlib import Path
import subprocess, sys, os
REPO = "https://github.com/baconkak/Chest-X-Ray-Pneumonia-COVID-19-Screening.git"
CODE_REF = "main"  # Replace with the saved commit SHA to reproduce this code version.
project = Path("/content/chest-xray-improved")
if not project.exists():
    subprocess.run(["git", "clone", REPO, str(project)], check=True)
os.chdir(project)
subprocess.run(["git", "fetch", "origin", CODE_REF], check=True)
subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], check=True)
sys.path.insert(0, str(project))
# A reused runtime may have imported src from the older notebook already.
for module in list(sys.modules):
    if module == "src" or module.startswith("src."):
        del sys.modules[module]
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "scikit-learn>=1.5,<2", "pandas>=2.2,<4", "matplotlib>=3.8,<4",
                "Pillow>=10.4,<13", "kagglehub>=0.3,<1", "tqdm>=4.66,<5"], check=True)
import torch
from src.engine import environment
CODE_SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print(environment())
print("Code commit:", CODE_SHA)
''')
cell(3, '''## 2. แผนการทดลองและที่เก็บผล
ค่าเริ่มต้น 3 งาน: ResNet18 weight=10, ResNet50 weight=10 และ ResNet50 weight=1
weight=10 เป็นค่าทดลองเพื่อลดความรุนแรงจากอัตราส่วนประมาณ 112 รอบก่อน ไม่ใช่ค่าที่พิสูจน์ว่าดีที่สุด
weight=1 เท่ากับ BCE ปกติ; ไม่มี oversampling ร่วมด้วย
RUN_EXTRA_ABLATIONS=True เพิ่ม ResNet50 weight=30 และอัตราส่วนจาก train โดยอัตโนมัติ
RUN_DENSENET=True เพิ่ม DenseNet121; แต่ละตัวเพิ่มเวลาฝึก

TRAIN_SEEDS=[42,43,44] ใช้ตรวจความแปรปรวนเมื่อมีเวลา; **SPLIT_SEED คงเดิม**
เลือก family ด้วยค่าเฉลี่ย validation AP และใช้ seed แรกตามแผน ไม่เลือก seed ที่คะแนนดีที่สุด
ต้องเปลี่ยน RUN_NAME ถ้าเปลี่ยนแผน/config; อย่า resume checkpoint รุ่นเก่า
ค่าเริ่มต้นสูงสุด 20 epochs, head 2 แล้ว fine-tune block สุดท้าย; patience 5
epoch แรกจะบอกความเร็วจริงเพื่อประมาณเวลาที่เหลือ งาน full นานกว่า quick อย่างชัดเจน
''')
cell(4, '''from dataclasses import replace
from src.engine import TrainConfig, save_json
USE_DRIVE = True
PROFILE = "full"
RUN_NAME = "nih_full_resnet50_v3"
RESUME = False
SPLIT_SEED = 42
TRAIN_SEEDS = [42]  # Optional: [42, 43, 44], declared before starting a new run.
RUN_EXTRA_ABLATIONS = False
RUN_DENSENET = False
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
OUTPUT_BASE = Path("/content/drive/MyDrive/ChestXrayProject") if USE_DRIVE else Path("outputs")
OUT = OUTPUT_BASE / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
CFG = TrainConfig(seed=TRAIN_SEEDS[0], epochs=20, head_epochs=2, batch_size=64,
                  lr=1e-4, finetune_lr=1e-5, patience=5, positive_weight=10.,
                  num_workers=4, amp=True, amp_init_scale=1024., progress=True)
SPLIT_MODE = "official" if PROFILE == "full" else "grouped"
MAX_PATIENTS = 3000 if PROFILE == "quick" else None
ALLOW_PARTIAL = False
if PROFILE not in {"full", "quick", "sample"}:
    raise ValueError("PROFILE must be full, quick or sample")
EXPERIMENTS = [
    dict(id="resnet18_pw10", model="resnet18", positive_weight=10.),
    dict(id="resnet50_pw10", model="resnet50", positive_weight=10.),
    dict(id="resnet50_pw1", model="resnet50", positive_weight=1.),
]
if RUN_EXTRA_ABLATIONS:
    EXPERIMENTS += [dict(id="resnet50_pw30", model="resnet50", positive_weight=30.),
                    dict(id="resnet50_pwauto", model="resnet50", positive_weight=None)]
if RUN_DENSENET:
    EXPERIMENTS += [dict(id="densenet121_pw10", model="densenet121", positive_weight=10.)]
print("Output:", OUT)
print("Training jobs:", len(EXPERIMENTS)*len(TRAIN_SEEDS))
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
if not torch.cuda.is_available():
    raise RuntimeError("Choose a GPU runtime before full-data training")
''')
cell(5, '''## 3. ใช้ NIH ชุดเต็มที่มีอยู่ หรือดาวน์โหลด
ข้อมูลเดิมที่แตกครบแล้วยังอยู่ใน runtime จะถูกใช้ต่อ ไม่ต้องดาวน์โหลดซ้ำ
runtime ใหม่ต้องโหลด archive ประมาณ 42 GB และแตกไฟล์ จึงต้องมี free disk อย่างน้อย 110 GiB
ควรใช้ดิสก์รวมประมาณ 210 GB ขึ้นไป และเก็บภาพใน /content ไม่อ่านภาพทีละไฟล์ผ่าน Drive
ไฟล์ต้นฉบับไม่ถูกลบ; cache ภาพเล็กสร้างแยกในขั้นตอนถัดไป
ถ้าเก็บภาพไว้ที่อื่น ตั้ง DOWNLOAD_DATA=False แล้วแก้ DATA_ROOT ให้ถูกต้อง
[แหล่งข้อมูล NIH บน Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
''')
cell(6, '''import shutil
DOWNLOAD_DATA = True
DATA_ROOT = Path("/content/nih-chestxray14")
KAGGLE_HANDLE = ("nih-chest-xrays/sample/versions/4" if PROFILE == "sample"
                 else "nih-chest-xrays/data/versions/3")
if DOWNLOAD_DATA:
    handle_name, version = ("sample", "4") if PROFILE == "sample" else ("data", "3")
    cached = Path.home()/".cache/kagglehub/datasets/nih-chest-xrays"/handle_name/"versions"/version
    metadata = list(cached.rglob("sample_labels.csv")) + list(cached.rglob("Data_Entry_2017*.csv"))
    if cached.is_dir() and len(metadata) == 1:
        DATA_ROOT = cached
        print("Using existing extracted dataset:", DATA_ROOT)
    else:
        import kagglehub
        free_gib = shutil.disk_usage("/content" if Path("/content").exists() else ".").free/2**30
        required = 10 if PROFILE == "sample" else 110
        print(f"Free disk: {free_gib:.1f} GiB")
        if free_gib < required:
            raise RuntimeError(f"Need at least {required} GiB free for download/extraction")
        DATA_ROOT = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
if not DATA_ROOT.is_dir():
    raise FileNotFoundError(DATA_ROOT)
print("Dataset path:", DATA_ROOT)
''')
# Keep metadata checks and immutable manifests from the original workflow.
nb.cells[8].source = nb.cells[8].source.replace('seed=CFG.seed', 'seed=SPLIT_SEED')
cell(9, '''## 5. Cache ภาพลง local disk และตรวจภาพ
ย่อแบบ pad + resize เป็น 224×224 **ครั้งเดียว** ก่อน augmentation แล้วบันทึก PNG RGB แบบไม่สูญเสียเพิ่ม
ลดการเปิดและย่อ PNG ต้นฉบับขนาดใหญ่ทุก epoch; ไม่ cache ภาพที่สุ่มหมุน/เลื่อนแล้ว
ชื่อ cache อิง path, ขนาดไฟล์, เวลาแก้ไขและรุ่น preprocessing; ใช้ต่อได้ใน runtime เดิม
เปลี่ยนข้อมูลต้นฉบับให้สร้าง cache ใหม่ (ระบบนี้ไม่ใช้ content hash ของภาพทุกไฟล์)
ขั้นตอนแรกยังใช้เวลา แต่ progress bar แสดงความคืบหน้า
ใช้ 4 workers + pinned memory; ปิด persistent workers เพื่อให้ resume สุ่ม augmentation ต่อได้ที่ขอบ epoch
Batch 64 เป็นค่าเริ่มต้นสำหรับ L4 ที่ 224 px; ถ้า OOM ให้เริ่ม RUN_NAME ใหม่ด้วย batch 32
GPU RAM ไม่จำเป็นต้องเต็ม และต้องดู img/s ร่วมด้วย
''')
original_view = nb.cells[10].source
cell(10, '''from src.data import prepare_image_cache
CACHE_IMAGES = True
CACHE_ROOT = Path("/content/nih_cache") if Path("/content").exists() else Path("data/nih_cache")
if CACHE_IMAGES:
    free_gib = shutil.disk_usage("/content" if Path("/content").exists() else ".").free/2**30
    if free_gib < 12:
        raise RuntimeError("Keep at least 12 GiB free for cache/checkpoints or set CACHE_IMAGES=False")
    combined = pd.concat(list(splits.values()), ignore_index=True)
    cached_frame = prepare_image_cache(combined, CACHE_ROOT, CFG.image_size, workers=4)
    paths = cached_frame.set_index("Image Index").path
    splits = {key: frame.assign(path=frame["Image Index"].map(paths)) for key, frame in splits.items()}
    save_json(dict(cache_root=str(CACHE_ROOT), size=CFG.image_size,
                   transform="RGB pad+resize v1; random affine remains online"), OUT/"cache_info.json")
''' + '\n' + original_view)
cell(11, '''## 6. ฝึกและเปรียบเทียบโมเดล
ResNet50 มีประมาณ 23.51 ล้านพารามิเตอร์หลังเปลี่ยนหัวเป็น 1 logit เทียบ ResNet18 ประมาณ 11.18 ล้าน
DenseNet121 ตัวเลือกมีประมาณ 6.95 ล้าน จึงลึกกว่าแต่ไม่ใหญ่กว่าในจำนวนพารามิเตอร์
ทุกโมเดล pretrained จาก ImageNet ไม่ใช่ pretrained สำหรับ COVID หรือ NIH
ฝึกหัวก่อน แล้วเปิด block สุดท้าย; BatchNorm ใช้สถิติ pretrained คงที่
LR หัว=0.0001 / fine-tune=0.00001; ลดลงอีกเมื่อ validation AP ไม่ดีขึ้น
ใช้ BCEWithLogitsLoss เพราะเป็น binary label เดียวและรับ logit โดยตรง; weight เทียบจาก validation
AMP เริ่ม scale=1024, clip norm=5 และตรวจ non-finite; ไม่ oversample พร้อม weighted loss
แสดง progress, img/s, peak GPU memory และเวลาราย epoch ใน history.csv

แต่ละ trial เก็บแยก; seed ของ split ไม่เปลี่ยนตาม seed training
ยังไม่เปิด final test ระหว่างเลือกโมเดลหรือน้ำหนัก loss
''')
cell(12, '''from src.engine import train_model, load_trained
from src.experiments import run_experiments, lock_selection
validation_table, seed_summary = run_experiments(
    splits, CFG, EXPERIMENTS, TRAIN_SEEDS, split_hash, OUT, resume=RESUME)
display(validation_table)
display(seed_summary)
display(pd.read_csv(OUT/"validation_operating_points.csv"))
''')
cell(13, '''## 7. ล็อกตัวเลือกจาก validation
เลือก experiment ด้วย mean AP ของ seeds ตามแผน; ถ้าเสมอใช้ลำดับที่ประกาศ
ใช้ checkpoint ของ seed แรกเสมอ ไม่เลือก seed ที่ได้คะแนนดีที่สุด
threshold หลัก: precision สูงสุดภายใต้ validation recall ≥ 0.90 เช่นเดิม
ตาราง operating points แสดง recall เป้าหมาย 0.50/0.80/0.90 และ threshold 0.5 เพื่อดู false positives
ตารางนี้เป็นการวิเคราะห์ validation; ไม่เปลี่ยนเกณฑ์หลักอัตโนมัติหลังเห็นผล
ถ้าอยากปรับเกณฑ์ ต้องตัดสินใจก่อนเปิด test และบันทึกแผนใหม่ให้ชัดเจน
''')
cell(14, '''selection = lock_selection(validation_table, seed_summary, TRAIN_SEEDS, split_hash, OUT)
winner = selection["selected_trial"]
save_json(dict(training_code_commit=CODE_SHA), OUT/"code_version.json")
print("Selected experiment:", selection["selected_experiment"])
print("Checkpoint:", OUT/winner/selection["selected_model"]/"best.pt")
print("Validation summary:", OUT/"validation_comparison.csv")
''')
cell(16, '''RUN_FINAL_TEST = False  # Freeze development before setting True.
if RUN_FINAL_TEST:
    from src.data import make_loaders
    from src.engine import predict
    from src.metrics import binary_metrics, patient_bootstrap_ap
    from src.plots import evaluation_plot
    test_dir = OUT/"final_test"
    if (test_dir/"STARTED.json").exists():
        raise RuntimeError("Test was already opened. Read saved artifacts; do not tune on test.")
    test_dir.mkdir(exist_ok=True)
    save_json(dict(selection=selection, code_commit=CODE_SHA), test_dir/"STARTED.json")
    loaders, _ = make_loaders(splits, CFG.batch_size, SPLIT_SEED, CFG.image_size, CFG.num_workers)
    test_frame = splits["test"]
    rows = []
    predictions = test_frame[["Image Index", "patient_id", "target"]].copy()
    for trial in selection["trials"]:
        trial_id, name = trial["trial"], trial["model"]
        model, _ = load_trained(OUT/trial_id/name/"best.pt", "cuda" if torch.cuda.is_available() else "cpu")
        y, scores = predict(model, loaders["test"], next(model.parameters()).device)
        metrics = binary_metrics(y, scores, selection["thresholds"][trial_id])
        predictions[trial_id] = scores
        boot = test_frame[["patient_id", "target"]].copy(); boot["score"] = scores
        ci = patient_bootstrap_ap(boot, repeats=1000)
        rows.append(dict(trial=trial_id, model=name, **metrics, ap_ci_low=ci["low"], ap_ci_high=ci["high"]))
        evaluation_plot(y, scores, metrics, test_dir/f"{trial_id}_evaluation.png")
        errors = test_frame.drop(columns="path").copy()
        errors["score"] = scores; errors["prediction"] = (scores >= metrics["threshold"]).astype(int)
        errors["error"] = np.select([(errors.target==1)&(errors.prediction==0),
                                     (errors.target==0)&(errors.prediction==1)], ["FN","FP"], default="correct")
        errors.to_csv(test_dir/f"{trial_id}_errors.csv", index=False)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    prevalence = float(splits["train"].target.mean())
    rows.append(dict(trial="train_prevalence_baseline", model="baseline",
                     **binary_metrics(y, np.full(len(y), prevalence), .5)))
    predictions.to_csv(test_dir/"all_predictions.csv", index=False)
    results = pd.DataFrame(rows); results.to_csv(test_dir/"test_comparison.csv", index=False)
    reference = f"resnet18_pw10_seed{TRAIN_SEEDS[0]}"
    delta = patient_bootstrap_ap(predictions, score_a=reference, score_b=winner, repeats=1000)
    save_json(dict(comparison=f"{winner} minus {reference} AP", paired_patient_bootstrap=delta),
              test_dir/"paired_ap_delta.json")
    display(results)
else:
    print("Training/validation completed. Final test is still unopened.")
''')
cell(17, '''## 9. ลองภาพเดี่ยวด้วยโมเดลที่เลือก และ Grad-CAM
ภาพ PNG/JPEG; ไม่ใช่ DICOM และไม่ใช่ตัววินิจฉัย
Grad-CAM ใช้ block สุดท้ายของ ResNet หรือ DenseNet และไม่ได้ยืนยันตำแหน่งรอยโรค
score จาก weighted BCE ยังไม่ใช่ probability ทางคลินิกที่ผ่าน calibration
''')
source = nb.cells[18].source.replace('from src.explain import grad_cam_resnet', 'from src.explain import grad_cam')
source = source.replace('OUT / "resnet18" / "best.pt"', 'OUT / winner / selection["selected_model"] / "best.pt"')
source = source.replace('selection["thresholds"]["resnet18"]', 'selection["thresholds"][winner]')
source = source.replace('grad_cam_resnet(model, x)', 'grad_cam(model, x)')
cell(18, source)
cell(19, '''## 10. ผลลัพธ์และการฝึกต่อ
Drive → MyDrive → ChestXrayProject → nih_full_resnet50_v3
ดู validation_comparison.csv, validation_seed_summary.csv, validation_operating_points.csv
แต่ละ trial มี run.json, history.csv, learning_curves.png, best.pt, last.pt และ val_predictions.csv
locked_selection.json บอกโมเดล/threshold ที่เลือก; final_test สร้างเมื่อเปิด test เท่านั้น
std_ap ว่างเมื่อมี seed เดียว เป็นค่าที่คำนวณไม่ได้ ไม่ใช่ความแปรปรวนศูนย์
save checkpoints ทุกขอบ epoch; หลุดกลาง epoch จะเริ่ม epoch นั้นใหม่
config ต้องตรงเดิม; เปลี่ยนโมเดล/batch/loss/seeds ต้องเริ่ม RUN_NAME ใหม่
อย่าเทียบ AP ตรงกับ quick รอบก่อน เพราะชุดผู้ป่วย/validation ต่างกัน ให้เทียบ baseline รอบ full เดียวกัน
ผลดีขึ้นหรือไม่ต้องพิจารณา AP, false positives, patient bootstrap และความแปรปรวนข้าม seeds ร่วมกัน
อ่านเหตุผลฉบับละเอียดใน README.md หัวข้อรุ่นปรับปรุง

อ้างอิง: [ResNet50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html),
[DenseNet121](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html),
[PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html),
[BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
''')
nb.metadata.colab = {'name': 'Chest_Xray_Improved_Colab.ipynb'}
for index, c in enumerate(nb.cells):
    c.id = f'improved-{index:02d}'
    if c.cell_type == 'code':
        c.outputs = []
        c.execution_count = None
nbf.validate(nb)
nbf.write(nb, ROOT/'Chest_Xray_Improved_Colab.ipynb')
print(f'Built improved notebook: {len(nb.cells)} cells')
