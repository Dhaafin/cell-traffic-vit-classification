# Command List Eksekusi Proyek

## 1. Setup Lingkungan
```bash
pip install -r requirements.txt
```

## 2. Quick Demo Inference
```bash
python demo_inference.py
```

## 3. Benchmark Keseluruhan (All Models)
```bash
# Validasi Temporal (Stratified 5-Fold)
python experiments/run_benchmark.py --model all --cv_mode stratified

# Validasi Spasial 19 Sektor eNodeB (Stratified Group 5-Fold)
python experiments/run_benchmark.py --model all --cv_mode group
```

## 4. Evaluasi Model Individual - Validasi Temporal (Stratified)
```bash
# DayViT (Ours)
python experiments/evaluate_cv_dayvit.py --cv_mode stratified

# ViT-Tiny Pretrained (LP-FT ImageNet-1k)
python experiments/evaluate_cv_vit_tiny_pretrained.py --cv_mode stratified

# ViT-Tiny From-Scratch (Random Initialization)
python experiments/evaluate_cv_vit_tiny_scratch.py --cv_mode stratified
```

## 5. Evaluasi Model Individual - Validasi Spasial (Group K-Fold)
```bash
# DayViT (Ours)
python experiments/evaluate_cv_dayvit.py --cv_mode group

# ViT-Tiny Pretrained (LP-FT ImageNet-1k)
python experiments/evaluate_cv_vit_tiny_pretrained.py --cv_mode group

# ViT-Tiny From-Scratch (Random Initialization)
python experiments/evaluate_cv_vit_tiny_scratch.py --cv_mode group
```
