 ### 3.1 Datasets

  All experiments are based on a Rocklin-style protein sequence–property dataset stored in data.csv. Each entry contains a protein identifier, a predefined train/validation/test split, a cluster assignment, an amino-acid sequence, and
  experimentally derived aggregation-related labels. In this work, we use log2_fold_change_75_clip as the primary anti-aggregation label.

  The full dataset contains 13,853 proteins, split into 9,774 training samples, 1,279 validation samples, and 2,800 test samples. At the cluster level, these correspond to 4,744, 659, and 1,351 clusters, respectively. Because inverse
  folding is performed on backbone structures rather than raw sequence rows, we first construct a cluster-balanced representative backbone set by sampling one protein per (split, cluster) group using a fixed random seed. This yields
  4,744 representative training backbones, 657 representative validation backbones, and 1,350 representative test backbones.

  To provide stability supervision, we merge the main Rocklin table with Metagenomic_dG.csv by protein identifier and obtain matched (\Delta G) labels. These labels are used both for training the stability predictor and for defining
  wild-type-relative stability improvement in downstream evaluation.

  For preference optimization, we further construct a threshold-balanced backbone subset from the representative set. Specifically, within each split, we retain all representative backbones whose aggregation label satisfies
  log2_fold_change_75_clip < -1.0, and then randomly sample an equal number of representatives from the complementary set. This produces a threshold-balanced subset containing 2,866 training backbones, 366 validation backbones, and 778
  test backbones. In the final reported experiments, we use the threshold-balanced subset for training and validation, but evaluate all final models on the full representative test set of 1,350 backbones.

  ———

  ### 3.2 Property Predictors

  To guide preference construction and candidate reranking, we train two backbone-conditioned property predictors: an anti-aggregation predictor and a stability predictor. Both predictors use the same SaProt-based sequence–structure
  representation and are trained independently before any preference optimization begins.

  Each protein is represented by its SaProt sequence with Foldseek-derived structural tokens (sa_sequence_foldseek). This representation is converted into the tokenized format required by the SaProt tokenizer and fed into a shared
  predictor architecture based on SaProt-650M with lightweight LoRA adaptation and an MLP prediction head (proagg_mlp_v36). LoRA is applied to the attention projections of the encoder, allowing efficient task adaptation while
  preserving the pretrained structural prior.

  #### Anti-aggregation predictor

  The anti-aggregation predictor is trained to regress log2_fold_change_75_clip. Training is performed on the original train/validation/test split of the Rocklin-derived dataset rather than on the DPO subset splits. The final model
  used throughout this work is selected by validation Spearman correlation. In addition to the primary regression objective, the final anti-aggregation model includes auxiliary regularization terms during training, including a ranking-
  oriented aggregation objective and a weak auxiliary (\Delta G) prediction term. The final checkpoint used in downstream experiments is version_92, which achieves test-set PCC 0.7462 and SCC 0.7562.

  #### Stability predictor

  The stability predictor is trained specifically to regress experimental (\Delta G). Again, training uses the original dataset split, restricted to samples for which a matched (\Delta G) label is available after merging with
  Metagenomic_dG.csv. The final model is selected by validation Spearman correlation. The final checkpoint used in all downstream experiments is version_96, which achieves test-set PCC 0.7406 and SCC 0.7065.

  These predictors are used only as learned scoring functions for preference construction and candidate ranking. They are not treated as the final measure of design quality; final model comparison is performed only after explicit
  structure validation.

  ———

  ### 3.3 Candidate Sampling and Preference Construction

  We optimize a structure-conditioned sequence generator under a fixed-backbone inverse-folding setting. For each backbone (x), we use ProteinMPNN as the sequence policy and sample multiple candidate sequences from the current policy.
  In the final experiments, we sample (N=16) sequences per backbone with temperature (T=0.5). Duplicate sequences are removed while preserving order.

  Each candidate sequence is evaluated by the two trained property predictors. To score a designed sequence under a fixed-backbone assumption, we combine the designed amino-acid sequence with the original backbone structural tokens to
  form a structure-aware SaProt input, and then apply the corresponding predictor. This yields an anti-aggregation score and a predicted stability value (\Delta G) for every sampled candidate.

  Before constructing preference pairs, we apply a wild-type-relative stability gate to exclude clearly unstable candidates. Let (\Delta G(y)) denote the predicted stability of candidate (y), and let (\Delta G_{\mathrm{WT}}) denote the
  stability of the wild-type protein associated with that backbone. We retain a candidate only if
  [
  \Delta G(y) \ge \Delta G_{\mathrm{WT}} - m,
  ]
  where (m=0.5) in the final experiments. This limits preference construction to candidates that remain within a bounded stability margin of the wild type.

  After filtering, we build preference pairs using a rank-aligned pairing strategy. Candidates are sorted in descending order, and the top half is paired with the bottom half. For joint preference construction, we sort candidates
  lexicographically by anti-aggregation score and then by predicted (\Delta G), both in descending order. A pair ((y_w, y_l)) is retained only if the preferred sequence (y_w) outperforms the dispreferred sequence (y_l) on both
  objectives:
  [
  s_{\mathrm{agg}}(y_w) - s_{\mathrm{agg}}(y_l) > \delta_{\mathrm{agg}},
  ]
  [
  s_{\mathrm{stab}}(y_w) - s_{\mathrm{stab}}(y_l) > \delta_{\mathrm{stab}},
  ]
  where (\delta_{\mathrm{agg}} = 0.20) and (\delta_{\mathrm{stab}} = 0.20) in the final setting. Thus, every retained preference pair encodes simultaneous improvement in anti-aggregation and stability.

  The same sampled pair set is used for all downstream training variants. In pure DPO, each pair contributes a winner–loser preference example. In pure SFT, only the winner sequence is used as a supervised target. In hybrid DPO+SFT,
  both the pairwise preference relation and the winner sequence likelihood are used jointly.

  ———

  ### 3.4 Optimization Objectives

  We study three training objectives on top of the same structure-conditioned ProteinMPNN policy.

  #### Pure DPO

  For a preference pair ((x, y_w, y_l)), pure DPO optimizes the policy (\pi_\theta) relative to a reference policy (\pi_{\mathrm{ref}}) using
  [
  \mathcal{L}_{\mathrm{DPO}}

  -\log \sigma \Big(
  \beta \big[
  (\log \pi_\theta(y_w|x) - \log \pi_{\mathrm{ref}}(y_w|x))

  (\log \pi_\theta(y_l|x) - \log \pi_{\mathrm{ref}}(y_l|x))
  \big]
  \Big),
  ]
  where (\beta) is the DPO temperature parameter. In the final experiments, (\beta = 0.1).

  #### Winner-only SFT

  To isolate the contribution of pairwise preference learning, we also train a winner-only supervised baseline:
  [
  \mathcal{L}_{\mathrm{SFT}}

  -\log \pi_\theta(y_w|x).
  ]
  This baseline uses the same sampled preference data and the same semi-online outer loop, but discards the loser and simply imitates preferred sequences.

  #### Hybrid DPO+SFT

  Finally, we consider a hybrid objective combining pairwise preference learning and winner-only supervised regularization:
  [
  \mathcal{L}_{\mathrm{hybrid}}

  \mathcal{L}{\mathrm{DPO}} + \lambda{\mathrm{SFT}} \mathcal{L}_{\mathrm{SFT}},
  ]
  where (\lambda{\mathrm{SFT}}) controls the strength of the supervised term. In our final comparisons, we evaluate several values of (\lambda_{\mathrm{SFT}}), and the best final structure-aware result is obtained with
  (\lambda_{\mathrm{SFT}} = 1.0).

  For all methods, sequence log-probabilities are computed by teacher-forcing the full target sequence and averaging conditional log-probabilities over designed positions. To reduce stochasticity in DPO scoring, we fix the decoding-
  order tensor within each training example, ensuring that policy and reference log-probabilities are computed under the same ordering.

  ———

  ### 3.5 Semi-Online Training Pipeline

  To reduce the mismatch between a changing policy and stale offline preference data, we use a semi-online training procedure. Instead of constructing one fixed preference dataset and optimizing on it to convergence, we iteratively
  refresh preference data from the current policy and perform only a short local update in each round.

  At the beginning of each round, the current ProteinMPNN checkpoint is used to resample candidate sequences for the training backbones, rescore them with the aggregation and stability predictors, and rebuild the preference dataset.
  The same process is repeated on the validation split to construct a round-specific validation set. Training is then performed for a small number of epochs using one of the three objectives above.

  In the final experiments, we use two semi-online rounds. To stabilize optimization further, we do not use the entire threshold-balanced training set in both rounds. Instead, we shuffle the training backbones with a fixed seed and
  partition them into two disjoint halves. The first half is used in round 1 and the second half is used in round 2. This design reduces per-round update pressure while still covering the full threshold-balanced training subset across
  rounds.

  In the DPO and hybrid settings, the selected checkpoint from the previous round initializes both the trainable policy and the frozen reference for the next round. Therefore, our DPO variants use a moving local reference rather than a
  single global reference fixed to the original ProteinMPNN checkpoint. In the SFT setting, the selected checkpoint simply initializes the next round’s trainable policy.

  Optimization is implemented with Adam and gradient clipping. Although examples are processed one at a time, we accumulate gradients across multiple steps to obtain an effective batch size of 32 in the final experiments.

  ———

  ### 3.6 Validation-Based Checkpoint Selection

  We do not use test performance for model selection. We also found that training-time validation loss alone is not a reliable proxy for downstream property improvement. Therefore, after each round, all saved epoch checkpoints are re-
  evaluated on a fixed threshold-balanced validation benchmark using downstream anti-aggregation and stability metrics.

  Checkpoint selection follows a tiered rule. Tier A requires non-negative validation improvement in both mean and best-of-(N) anti-aggregation and stability:
  [
  \Delta \mathrm{Agg}_{\mathrm{mean}} \ge 0,\quad
  \Delta \mathrm{Agg}{\max} \ge 0,\quad
  \Delta \Delta G_{\mathrm{mean}} \ge 0,\quad
  \Delta \Delta G_{\max} \ge 0.
  ]
  Among checkpoints satisfying Tier A, we choose the one with the highest validation joint score
  [
  \Delta \mathrm{Agg}{\mathrm{mean}} + \Delta \Delta G{\mathrm{mean}}.
  ]

  If no checkpoint satisfies Tier A, we fall back to Tier B, which requires only non-negative mean improvements for both objectives. If no checkpoint satisfies Tier B, we use a final Tier C fallback that ranks checkpoints by the same
  joint score with an additional penalty for negative max metrics. The selected checkpoint is used as the starting policy for the next round and for final test evaluation.

  ———

  ### 3.7 Candidate Reranking and Chai-1 Structure Validation

  After training, final evaluation is performed on the full representative test set. For each test backbone, we sample 16 candidate sequences from the selected policy and annotate each sequence with four quantities: predicted anti-
  aggregation score, predicted (\Delta G), stability improvement relative to wild type (\Delta G - \Delta G_{\mathrm{WT}}), and ProteinMPNN log-probability.

  Because structure prediction is substantially more expensive than predictor-based scoring, we do not pass all candidates to Chai-1. Instead, for each backbone we retain a top-3 candidate set using staged filtering and reranking. We
  prioritize candidates that pass pathology filtering and satisfy both the anti-aggregation and stability criteria, then fall back progressively to weaker filtering stages if necessary. Within each stage, candidates are ranked by
  descending anti-aggregation score, then descending (\Delta G - \Delta G_{\mathrm{WT}}), and finally descending ProteinMPNN log-probability.

  The top-3 candidates for each backbone are then evaluated with Chai-1. Since Chai-1 may produce multiple structure samples for one candidate, we select a single best predicted structure per candidate according to the confidence-based
  ranking used in our evaluation pipeline, and record candidate-level structure confidence metrics including pLDDT and pTM.

  ———

  ### 3.8 Final Metrics

  All final results are summarized at the backbone level. For each backbone, we compute the mean over its top-3 retained candidates for the following quantities: anti-aggregation score, (\Delta G - \Delta G_{\mathrm{WT}}), ProteinMPNN
  log-probability, best-structure pLDDT, and best-structure pTM. We refer to this protocol as backbone-level mean-of-top3 evaluation.

  Wild-type aggregation references are taken directly from the experimental Rocklin dataset using log2_fold_change_75_clip, and wild-type stability references are taken from Metagenomic_dG.csv. These wild-type references define
  relative improvement for both sequence-level filtering and final evaluation.

  Our primary headline metric is a strict joint success criterion. A candidate is counted as successful only if it simultaneously satisfies all of the following:
  [
  \text{predicted anti-aggregation score} > \text{wild-type aggregation label},
  ]
  [
  \Delta G > \Delta G_{\mathrm{WT}},
  ]
  [
  \text{pLDDT} > 0.85,
  ]
  [
  \text{pTM} > 0.80.
  ]
  We then aggregate this success indicator over the top-3 candidates of each backbone and report the resulting backbone-level mean-of-top3 strict joint success rate. This metric serves as our final structure-aware comparison criterion
  across all methods.