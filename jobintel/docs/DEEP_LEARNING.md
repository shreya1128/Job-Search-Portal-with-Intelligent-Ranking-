# Deep learning in the project (Phase 5)

This document explains **what was built, why, the mathematics, how to train and run it, and where it is used in the platform**.
All measured numbers live in [`phase5_report.md`](phase5_report.md) (generated from the saved result files); this file only states results that are structural (e.g. parameter counts).
Framework: **TensorFlow/Keras** throughout. No BERT/Transformers; no TF-IDF/BM25 as a primary retrieval method.

## 1. Where deep learning sits in the pipeline

```
 job data ──► preprocessing ──► DL tokenizer ──► sequence models (RNN / BiRNN / LSTM / GRU) ──► job-function classifier
 (Phase 1)    (Phase 1)         (jobintel/dl/tokenizer.py)        │                                     │
                                                                  └─► recurrent representation ──► DLEncoder.embed() ──► DL-similarity signal (Phase 6 hybrid recommender)
 skills of a job/resume ──► multi-hot vector ──► denoising autoencoder ──► suggest missing skills · skill-latent code (jobintel/dl/autoencoder.py)
 resume text / scans ──► (Phase 6) NER + document/image processing + OCR
```

| Syllabus topic | Where it is implemented / explained |
|---|---|
| Sequence learning, unfolding | §2, `jobintel/dl/numpy_rnn.py` (explicit loop over time) |
| RNN, BiRNN | §3, `jobintel/dl/models.py` (`kind="rnn"`, `"birnn"`), `numpy_rnn.rnn_forward/birnn_forward` |
| Vanishing / exploding gradients | §4, `jobintel/dl/gradients.py`, `scripts/run_gradient_experiment.py`, report §3 |
| LSTM (gates, cell state, selective write/forget/read) | §5, `numpy_rnn.lstm_forward`, `jobintel/dl/gate_demo.py`, `scripts/trace_gates.py`, report §4 |
| GRU | §6, `numpy_rnn.gru_forward` |
| Comparison table RNN / BiRNN / LSTM / GRU | report §2 (accuracy, precision, recall, F1, training time, validation loss) |
| Autoencoders: basic, deep, **denoising (main)**, regularisation | §8, `jobintel/dl/autoencoder.py`, report §5 |
| Convolutional / variational autoencoders, image compression | §9, `jobintel/dl/case_studies.py`, report §6 |
| Network-traffic anomaly case study (kept separate) | §9, synthetic data, report §6 |
| DL project pipeline, saving models, no retraining per search | §7, §10 |
| NER, resume analyzer (document/image processing, OCR) | **Phase 6** (not part of this phase) |
| Image captioning, facial recognition, self-driving | brief text only, §11 |

## 2. Sequence learning and unfolding

A job description is a sequence of words `x_1 … x_T`. A recurrent network reads it one word at a time and keeps a **state** `h_t` that summarises what has been read:

```
h_t = f(h_{t-1}, x_t ; θ)          h_0 = 0
```

*Unfolding* draws this loop as a chain of T copies of the same function with **shared parameters θ**. Backpropagation through the unfolded chain is *backpropagation through time* (BPTT): the gradient of the loss with respect to an early input is a product of T−t Jacobians `∂h_{k+1}/∂h_k` (§4). Sharing θ across time is what lets one model handle any length; the product is what makes long sequences hard.

In this project the text is converted by one persisted object (`DLTokenizer`): lower-case words, numbers → `<num>`, vocabulary from the *training split only* (padding 0, unknown 1), truncated/padded to `max_len = 256`. An `Embedding(64)` layer with `mask_zero=True` makes the recurrence skip padding, so the final state is the state after the **last real word** (tested: padding does not change the output).

## 3. RNN and BiRNN

```
SimpleRNN :  h_t = tanh( W x_t + R h_{t-1} + b )
BiRNN     :  forward RNN reads x_1…x_T, a second RNN reads x_T…x_1;   output = [ h_T(fwd) ; h_1(bwd) ]
classifier:  p = softmax( Dense( Dropout( h ) ) )       loss = cross-entropy (balanced class weights)
```

Why a BiRNN: the state at the end of a one-directional pass is dominated by the last words; the backward pass gives the first words a short path to the output. The cost is twice the recurrent parameters and sequential time.

Common skeleton for all four models (so the comparison isolates the recurrent cell):
`Embedding(64, masked) → recurrent layer (64 units) → Dropout(0.3) → Dense(softmax)`; Adam, lr 1e-3, `clipnorm = 1.0`; early stopping on validation loss (patience 2, best weights restored).
Parameter counts (from the architecture): SimpleRNN 833,863 · BiRNN 842,567 · LSTM 858,631 · GRU 850,567 (the 12,893-word embedding table is ≈ 825k of each).

## 4. Vanishing and exploding gradients

For the loss `L` at the last step,

```
∂L/∂x_t = ∂L/∂h_T · ( ∏_{k=t}^{T-1} ∂h_{k+1}/∂h_k ) · ∂h_t/∂x_t ,      ∂h_{k+1}/∂h_k = diag(f'(a_{k+1})) · R
```

so `‖∂L/∂x_t‖` is bounded by `(γ·σ_max(R))^{T−t}` times the rest, where γ bounds `|f'|` (1 for tanh) and σ_max is the largest singular value of R.
If `γ·σ_max < 1` the product **vanishes** (early words get no learning signal); if the recurrent map expands directions, it can **explode** (unstable updates, NaN).

`scripts/run_gradient_experiment.py` measures this with `tf.GradientTape` on bare layers: the gradient norm at every time step, T ∈ {10, 25, 50, 100, 200}, recurrent kernel scaled ×0.5/1/1.5/2, SimpleRNN (tanh and ReLU), LSTM and GRU (Keras defaults and with gates biased open), medians over 5 initialisations in float64 (report §3). It also shows the effect of clipping:

```
g ← g · min(1, τ / ‖g‖)        (global-norm clipping, τ = 1.0 in every model of the project)
```

Clipping keeps the direction and bounds the step; it cures *explosion*, not *vanishing*. Vanishing is addressed by the gated cells below.

## 5. LSTM

```
i_t = σ( W_i x_t + R_i h_{t-1} + b_i )      input gate  – how much NEW information to write
f_t = σ( W_f x_t + R_f h_{t-1} + b_f )      forget gate – how much of the OLD cell state to keep
g_t = tanh( W_c x_t + R_c h_{t-1} + b_c )   candidate   – what would be written
o_t = σ( W_o x_t + R_o h_{t-1} + b_o )      output gate – how much of the cell to read out
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t             cell state  – the additive memory path
h_t = o_t ⊙ tanh( c_t )
```

*Selective write / forget / read*: `i_t` decides what enters memory, `f_t` what stays, `o_t` what is exposed. Because `c_t` is updated **additively**, `∂c_t/∂c_{t−1} = diag(f_t)` (plus gate terms): when `f_t ≈ 1` the gradient passes through unchanged. Keras initialises the forget bias to 1 (σ(1) ≈ 0.73), which still shrinks the gradient over long spans; the gradient experiment shows that a larger forget bias keeps it alive, i.e. **the benefit comes from the gate being open, not from the cell existing**.

`numpy_rnn.py` implements these equations from scratch with the Keras weight layout (gate order i, f, c, o). `tests/test_dl_numpy_rnn.py` checks equality with the Keras layer (< 1e-5), masking, gate ranges and the additive-memory property.
`scripts/trace_gates.py` shows the gates on a real job description with the trained model; `scripts/run_gate_demo.py` uses a small **synthetic** memory task (marker → value → noise → report the value) where selective write/keep/read is required and therefore visible. The report states honestly that on real text the *average* gates move only slightly.

## 6. GRU

```
z_t = σ( W_z x_t + R_z h_{t-1} + b_z )            update gate
r_t = σ( W_r x_t + R_r h_{t-1} + b_r )            reset gate
n_t = tanh( W_h x_t + r_t ⊙ (R_h h_{t-1} + b_h) )  candidate (Keras reset_after = True)
h_t = z_t ⊙ h_{t-1} + (1 − z_t) ⊙ n_t
```

Two gates and no separate cell state: fewer parameters than an LSTM and a similar additive path (`z_t` plays the role of the forget gate). Implemented in `numpy_rnn.gru_forward` and verified against Keras.

## 7. Training documentation (job-function classification)

* **Task.** Predict a job's *function* from its description (EMSCAD `function` label; classes with ≥ 250 jobs; 'Other' dropped). **The title is not an input.**
* **Leakage control.** Near-duplicate descriptions (SimHash) are collapsed *before* splitting; the split is **by employer** (stratified): train / validation / test with no employer in two splits.
* **Class imbalance.** Balanced class weights `w_c = N / (K · n_c)` in the loss; macro-averaged precision/recall/F1 are reported next to accuracy.
* **Early stopping** on validation loss; the deployed model is chosen by **validation** loss, never by test results.
* **Comparison table** (report §2): accuracy, macro precision, macro recall, macro F1, validation loss, training time, epochs, parameters, inference ms/job — mean and range over seeds, plus a non-neural reference (binary bag-of-words + logistic regression).
* **Resumable training**: `python scripts/train_dl.py --model baseline rnn birnn lstm gru --seeds 42 43 --skip-existing [--budget SECONDS]`. Results → `data/dl/results/*.json`, models → `data/dl/models/*.keras`, tokenizer → `data/dl/tokenizer.json`.
* **Never retrained per search.** Models and the tokenizer are loaded from disk (`jobintel/dl/encoder.py`); search and recommendation only call `predict`.
* **Reproducibility.** Seeds fix initialisation/shuffling; TensorFlow is not bit-reproducible across machines, so numbers can differ slightly between environments (documented in the report).

## 8. Autoencoders on skill vectors (denoising = main)

A job or resume is a set of skills → a multi-hot vector `x ∈ {0,1}^F` over the F = 1,500 most frequent non-ambiguous skills (vocabulary fixed on the training split). Real skill lists are incomplete and noisy, so the **denoising autoencoder (DAE)** is the main model:

```
corrupt:   x̃ = (x with each present skill dropped with p = 0.3, ≥ 1 dropped / ≥ 1 kept)  +  0…3 random extra skills
encode:    z = enc(x̃)  ∈ R^64          decode:  x̂ = σ( dec(z) ) ∈ (0,1)^F
loss:      BCE(x, x̂) = − 1/F Σ_j [ x_j log x̂_j + (1 − x_j) log(1 − x̂_j) ]       ← target is the CLEAN x
```

| Model | Architecture | Trained on |
|---|---|---|
| basic | F → 64 → F | clean input |
| deep | F → 512 → 256 → 64 → 256 → 512 → F | clean input |
| **denoising** | deep architecture | corrupted input → clean target (fresh noise every epoch) |
| *_noreg | same without L2 and dropout | ablation |

**Regularisation:** L2 penalty `λ Σ‖W‖²` (λ = 1e-5) and dropout 0.2 in the hidden layers; the ablation shows what happens without them. Corruption is itself a regulariser.
**Uses in the platform:** (a) `suggest_skills()` — given a user's skills, the skills they probably also have/need; (b) the 64-d latent code as a skill-profile embedding for jobs without description text (`scripts/build_dl_embeddings.py`); (c) a cleaned skill profile.
**Evaluation** (report §5): reconstruction BCE; cosine to the clean vector; recovery@k of the *dropped* skills against random, popularity and co-occurrence baselines (with an employer-clustered bootstrap for paired differences); AUC for separating true from randomly added skills; k-NN category purity of raw vs latent vectors.

## 9. Case studies (separate from the job workflow)

* **Image compression** – a convolutional autoencoder (Conv/MaxPool encoder → code of k numbers → UpSampling/Conv decoder) vs PCA at the same code size on scikit-learn's 8×8 digits. Compression ratio = 64/k; error = MSE/PSNR.
* **Variational autoencoder** – the encoder outputs `μ, log σ²`; `z = μ + σ ⊙ ε, ε ~ N(0, I)`; loss = reconstruction + β·KL(q(z|x) ‖ N(0, I)) with `KL = −½ Σ (1 + log σ² − μ² − σ²)`. Because the latent space is regularised towards N(0, I), images can be *sampled* from the prior; quality is checked with a classifier trained on real digits.
* **Network-traffic anomaly detection** – an autoencoder trained on **normal flows only**; a flow is flagged when its reconstruction error exceeds the 99th percentile of held-out normal flows. **The traffic is synthetic** (CIC-IDS2017 was unavailable); `load_cic_ids()` reads the real CSVs if supplied. Precision/recall/F1/AUC are reported with PCA and Isolation Forest baselines. Results describe the generator, not real intrusion-detection performance.

## 10. Integration with the rest of the project

* `DLEncoder` (`jobintel/dl/encoder.py`): loads the sequence model with the lowest validation loss, `embed(texts)` (L2-normalised recurrent state), `predict_category(texts)`. Empty/unusable text returns a zero vector / `None`, never an exception.
* `scripts/build_dl_embeddings.py`: embeddings for every job with description text (`data/dl/embeddings_text.npz`) and denoising-AE latent codes for every job with skills (`embeddings_skills.npz`), keyed by (`source`, `external_id`). The two spaces are never mixed; jobs without text use the skill space.
* Phase 6 uses them for the "DL similarity" signal of the 11-signal hybrid recommender (a signal that is unavailable for a job is dropped and the weights re-normalised, as in the IR ranker) and for user/resume representations.

## 11. Brief case-study notes (text only; not application modules)

* **Image captioning.** A convolutional encoder turns the image into a feature vector that initialises or conditions an LSTM/GRU decoder, which emits the caption word by word, each step fed by the previous word. It is the recurrent machinery of §3–6 with the image as the initial context; evaluation compares generated captions with references (BLEU-style n-gram overlap).
* **Facial recognition.** A convolutional network maps a face to an embedding where images of the same person are close; *verification* thresholds the distance between two embeddings, *identification* looks up the nearest stored embedding. Training uses metric-learning losses (contrastive/triplet). The same embed-then-compare idea is used here for job and resume similarity.
* **Self-driving.** Perception (convolutional detection/segmentation), prediction (recurrent models over object tracks) and planning/control are combined; safety is dominated by rare situations, sensor fusion and validation, which a classification accuracy does not capture.

## 12. Limitations

See report §7. In short: the recurrent models do not beat the bag-of-words baseline on this data; two seeds per model; descriptions truncated at 256 tokens; TensorFlow runs are not bit-reproducible; the denoising autoencoder ties (does not beat) a co-occurrence baseline; the memory task and the traffic data are synthetic; NER and the resume analyzer belong to Phase 6.
